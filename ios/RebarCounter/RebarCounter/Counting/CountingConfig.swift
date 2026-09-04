import Foundation

/// The handful of numbers the counting code needs, kept in one place.
///
/// `counting_config.json` is written into the bundle by python/ml/export_coreml.py
/// straight from config.yaml, so the phone and the desktop pipeline can never drift
/// apart silently. The literals below are only the last-resort fallback.
struct CountingConfig: Codable {
    var tileSize: Int = 640
    /// Photos larger than this are scaled down before tiling; see CountingService.
    var maxLongEdge: Int = 4032
    var overlap: Double = 0.20
    var edgeMarginRatio: Double = 0.10
    var iouThreshold: Double = 0.35
    var scoreThreshold: Double = 0.25
    var maxDetectionsPerTile: Int = 1200
    var modelName: String = "RebarDetector"

    static let `default` = CountingConfig()

    /// Decoded key by key, each one falling back to its default.
    ///
    /// Swift's synthesised Decodable treats every property as required even when it
    /// has a default value, so adding one field to counting_config.json would make
    /// every older file fail to decode as a whole - and the app would silently drop
    /// back to the built-in numbers instead of honouring the ones that are there.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = CountingConfig()
        tileSize = try container.decodeIfPresent(Int.self, forKey: .tileSize) ?? fallback.tileSize
        maxLongEdge = try container.decodeIfPresent(Int.self, forKey: .maxLongEdge)
            ?? fallback.maxLongEdge
        overlap = try container.decodeIfPresent(Double.self, forKey: .overlap) ?? fallback.overlap
        edgeMarginRatio = try container.decodeIfPresent(Double.self, forKey: .edgeMarginRatio)
            ?? fallback.edgeMarginRatio
        iouThreshold = try container.decodeIfPresent(Double.self, forKey: .iouThreshold)
            ?? fallback.iouThreshold
        scoreThreshold = try container.decodeIfPresent(Double.self, forKey: .scoreThreshold)
            ?? fallback.scoreThreshold
        maxDetectionsPerTile = try container.decodeIfPresent(
            Int.self, forKey: .maxDetectionsPerTile
        ) ?? fallback.maxDetectionsPerTile
        modelName = try container.decodeIfPresent(String.self, forKey: .modelName)
            ?? fallback.modelName
    }

    init() {}

    static func loadFromBundle(_ bundle: Bundle = .main) -> CountingConfig {
        guard
            let url = bundle.url(forResource: "counting_config", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let config = try? JSONDecoder().decode(CountingConfig.self, from: data)
        else {
            return .default
        }
        return config
    }
}
