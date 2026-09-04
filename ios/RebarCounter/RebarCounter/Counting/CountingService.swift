import CoreImage
import CoreML
import UIKit
import Vision

enum CountingError: LocalizedError {
    case modelMissing(String)
    case badImage
    case unsupportedOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            "Sayım modeli (\(name)) uygulamaya eklenmemiş. python/ml/export_coreml.py çalıştırılmalı."
        case .badImage:
            "Fotoğraf okunamadı."
        case .unsupportedOutput(let detail):
            "Model çıktısı beklenen biçimde değil: \(detail)"
        }
    }
}

/// Runs the detector over a whole photo, tile by tile, entirely on the device.
///
/// SPEC 2: nothing leaves the phone. There is no network code in this file and there
/// is not meant to be one.
actor CountingService {
    private let config: CountingConfig
    private var model: MLModel?
    private var inputSize: CGSize = CGSize(width: 640, height: 640)
    private var inputFeatureName: String = "image"

    init(config: CountingConfig = .loadFromBundle()) {
        self.config = config
    }

    var currentConfig: CountingConfig { config }

    /// True when a model is bundled, so the UI can say something useful instead of
    /// failing at the shutter button.
    static func isModelAvailable(config: CountingConfig = .loadFromBundle()) -> Bool {
        Bundle.main.url(forResource: config.modelName, withExtension: "mlmodelc") != nil
    }

    private func loadModelIfNeeded() throws -> MLModel {
        if let model { return model }
        guard let url = Bundle.main.url(forResource: config.modelName, withExtension: "mlmodelc") else {
            throw CountingError.modelMissing(config.modelName)
        }
        let configuration = MLModelConfiguration()
        // Neural Engine when there is one, GPU otherwise - never the network.
        configuration.computeUnits = .all
        let loaded = try MLModel(contentsOf: url, configuration: configuration)

        if let input = loaded.modelDescription.inputDescriptionsByName.first(where: {
            $0.value.type == .image
        }) {
            inputFeatureName = input.key
            if let constraint = input.value.imageConstraint {
                inputSize = CGSize(width: constraint.pixelsWide, height: constraint.pixelsHigh)
            }
        }
        model = loaded
        return loaded
    }

    /// Count one photo. `progress` is called with 0...1 so a 40-tile photo does not
    /// look frozen.
    func count(image: UIImage, progress: (@Sendable (Double) -> Void)? = nil) async throws -> [Detection] {
        guard let cgImage = image.cgImage else { throw CountingError.badImage }
        let model = try loadModelIfNeeded()

        let width = cgImage.width
        let height = cgImage.height
        let tiles = Tiler.plan(width: width, height: height, config: config)
        let margin = Tiler.edgeMargin(config)

        var merged: [Tiler.Box] = []
        for (index, tile) in tiles.enumerated() {
            guard let crop = cgImage.cropping(to: tile.rect) else { continue }
            guard let buffer = pixelBuffer(from: crop, size: inputSize) else { continue }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                inputFeatureName: MLFeatureValue(pixelBuffer: buffer)
            ])
            let output = try Self.predict(model, input)

            // The crop was stretched to the model's input square; undo that here.
            let scaleX = tile.rect.width / inputSize.width
            let scaleY = tile.rect.height / inputSize.height

            var boxes = try decode(output, scaleX: scaleX, scaleY: scaleY)
            boxes = Tiler.nms(boxes, iouThreshold: config.iouThreshold)
            if boxes.count > config.maxDetectionsPerTile {
                boxes = Array(boxes.sorted { $0.score > $1.score }.prefix(config.maxDetectionsPerTile))
            }
            for box in boxes where Tiler.keep(box: box.rect, in: tile, margin: margin) {
                merged.append(Tiler.Box(rect: Tiler.toGlobal(box.rect, tile: tile), score: box.score))
            }
            progress?(Double(index + 1) / Double(max(tiles.count, 1)))
        }

        return Tiler.detections(from: Tiler.nms(merged, iouThreshold: config.iouThreshold))
    }

    /// Synchronous prediction in a non-async context.
    ///
    /// Inside an async function the compiler picks CoreML's async overload, which
    /// would hop per tile - forty hops for one photo. Here the sync one is chosen.
    private nonisolated static func predict(
        _ model: MLModel,
        _ input: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        try model.prediction(from: input, options: MLPredictionOptions())
    }

    // MARK: - Model output

    /// Handles both shapes ultralytics can export: the raw `[1, 4 + classes, anchors]`
    /// tensor, and the NMS pipeline's `confidence` / `coordinates` pair.
    private func decode(_ output: MLFeatureProvider, scaleX: CGFloat, scaleY: CGFloat) throws -> [Tiler.Box] {
        if let confidence = output.featureValue(for: "confidence")?.multiArrayValue,
           let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue {
            return decodePipeline(confidence: confidence, coordinates: coordinates, scaleX: scaleX, scaleY: scaleY)
        }

        guard
            let name = output.featureNames.first(where: {
                output.featureValue(for: $0)?.multiArrayValue != nil
            }),
            let array = output.featureValue(for: name)?.multiArrayValue
        else {
            throw CountingError.unsupportedOutput("çıktı bulunamadı")
        }
        return try decodeRaw(array, scaleX: scaleX, scaleY: scaleY)
    }

    private func decodeRaw(_ array: MLMultiArray, scaleX: CGFloat, scaleY: CGFloat) throws -> [Tiler.Box] {
        // Expected [1, 4 + classes, anchors]; with one class that is [1, 5, anchors].
        let shape = array.shape.map(\.intValue)
        guard shape.count == 3, shape[1] >= 5 else {
            throw CountingError.unsupportedOutput("shape \(shape)")
        }
        let attributes = shape[1]
        let anchors = shape[2]
        let strides = array.strides.map(\.intValue)

        // A float16-quantised export can hand back a float16 output; reading it as
        // Float would be garbage rather than an error, so the type is checked here.
        let read: (Int) -> Double
        switch array.dataType {
        case .float32:
            let base = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            read = { Double(base[$0]) }
        case .float16:
            let base = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            read = { Double(base[$0]) }
        case .double:
            let base = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
            read = { base[$0] }
        default:
            throw CountingError.unsupportedOutput("dtype \(array.dataType.rawValue)")
        }

        func value(_ attribute: Int, _ anchor: Int) -> Double {
            read(attribute * strides[1] + anchor * strides[2])
        }

        // Ultralytics emits box coordinates in input pixels; some exports normalise
        // them to 0...1. Decide once from the centres, which span the whole tile in
        // pixel mode and can never all be under 1.5 there.
        var maximum = 0.0
        for anchor in stride(from: 0, to: anchors, by: Swift.max(1, anchors / 128)) {
            maximum = Swift.max(maximum, Swift.max(value(0, anchor), value(1, anchor)))
        }
        let unit = maximum <= 1.5 ? Double(inputSize.width) : 1.0

        var boxes: [Tiler.Box] = []
        boxes.reserveCapacity(256)
        for anchor in 0..<anchors {
            var score = value(4, anchor)
            for attribute in 5..<attributes {
                score = Swift.max(score, value(attribute, anchor))
            }
            guard score >= config.scoreThreshold else { continue }

            let cx = value(0, anchor) * unit
            let cy = value(1, anchor) * unit
            let w = value(2, anchor) * unit
            let h = value(3, anchor) * unit
            boxes.append(
                Tiler.Box(
                    rect: CGRect(
                        x: (cx - w / 2) * scaleX,
                        y: (cy - h / 2) * scaleY,
                        width: w * scaleX,
                        height: h * scaleY
                    ),
                    score: score
                )
            )
        }
        return boxes
    }

    private func decodePipeline(
        confidence: MLMultiArray,
        coordinates: MLMultiArray,
        scaleX: CGFloat,
        scaleY: CGFloat
    ) -> [Tiler.Box] {
        let count = confidence.shape[0].intValue
        let classes = confidence.shape.count > 1 ? confidence.shape[1].intValue : 1
        var boxes: [Tiler.Box] = []
        boxes.reserveCapacity(count)

        for index in 0..<count {
            var score = 0.0
            for klass in 0..<classes {
                score = max(score, confidence[[index, klass] as [NSNumber]].doubleValue)
            }
            guard score >= config.scoreThreshold else { continue }

            // The pipeline reports normalised centre/size.
            let cx = coordinates[[index, 0] as [NSNumber]].doubleValue * Double(inputSize.width)
            let cy = coordinates[[index, 1] as [NSNumber]].doubleValue * Double(inputSize.height)
            let w = coordinates[[index, 2] as [NSNumber]].doubleValue * Double(inputSize.width)
            let h = coordinates[[index, 3] as [NSNumber]].doubleValue * Double(inputSize.height)
            boxes.append(
                Tiler.Box(
                    rect: CGRect(
                        x: (cx - w / 2) * scaleX,
                        y: (cy - h / 2) * scaleY,
                        width: w * scaleX,
                        height: h * scaleY
                    ),
                    score: score
                )
            )
        }
        return boxes
    }

    // MARK: - Pixel buffers

    private func pixelBuffer(from image: CGImage, size: CGSize) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        guard
            CVPixelBufferCreate(
                kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                attributes as CFDictionary, &buffer
            ) == kCVReturnSuccess,
            let pixelBuffer = buffer
        else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue
            )
        else { return nil }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }
}
