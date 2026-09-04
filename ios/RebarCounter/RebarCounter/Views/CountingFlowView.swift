import SwiftData
import SwiftUI

/// Drives SPEC 3 end to end: photo -> count -> correct -> save.
///
/// The clock starts the moment the photo exists and stops on save, so the phase 5
/// "how long did it take" number is measured, not remembered.
struct CountingFlowView: View {
    enum Source {
        case camera
        case image(UIImage)
    }

    let source: Source
    var onFinished: () -> Void

    @Environment(\.modelContext) private var modelContext

    @State private var stage: Stage = .capture
    @State private var startedAt = Date()
    @State private var progress: Double = 0
    @State private var errorMessage: String?

    private let service = CountingService()

    private enum Stage {
        case capture
        case counting(UIImage)
        case result(UIImage, [Detection], Int)
    }

    var body: some View {
        Group {
            switch stage {
            case .capture:
                CameraView(
                    onCapture: { begin(with: $0) },
                    onCancel: onFinished
                )
            case .counting(let image):
                CountingProgressView(image: image, progress: progress)
            case .result(let image, let detections, let automatic):
                ResultView(
                    image: image,
                    detections: detections,
                    automaticCount: automatic,
                    onSave: { corrected, details in save(image, corrected, automatic, details) },
                    onRetake: { stage = .capture }
                )
            }
        }
        .task {
            if case .image(let image) = source, case .capture = stage {
                begin(with: image)
            }
        }
        .alert("Sayım yapılamadı", isPresented: .constant(errorMessage != nil)) {
            Button("Tamam") { errorMessage = nil; onFinished() }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func begin(with image: UIImage) {
        startedAt = Date()
        progress = 0
        stage = .counting(image)
        Task {
            do {
                let detections = try await service.count(image: image) { value in
                    Task { @MainActor in progress = value }
                }
                stage = .result(image, detections, detections.count)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func save(_ image: UIImage, _ detections: [Detection], _ automatic: Int, _ details: SaveDetails) {
        do {
            let fileName = try PhotoStore.save(image)
            let record = CountRecord(
                deliveryNote: details.deliveryNote,
                diameterMM: details.diameterMM,
                note: details.note,
                count: detections.count,
                automaticCount: automatic,
                correctionCount: abs(detections.count - automatic),
                photoFileName: fileName,
                points: detections,
                elapsedSeconds: Date().timeIntervalSince(startedAt),
                manualCount: details.manualCount,
                manualSeconds: details.manualSeconds
            )
            modelContext.insert(record)
            try modelContext.save()
            onFinished()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

private struct CountingProgressView: View {
    let image: UIImage
    let progress: Double

    var body: some View {
        ZStack {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .blur(radius: 6)
                .overlay(Color.black.opacity(0.45))
                .ignoresSafeArea()

            VStack(spacing: 14) {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 220)
                    .tint(.white)
                Text("Sayılıyor…")
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("Fotoğraf karolara bölünüp tek tek taranıyor")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.75))
            }
        }
    }
}
