import Foundation

/// The handful of numbers the counting code needs, kept in one place.
///
/// `counting_config.json` is written into the bundle by python/ml/export_coreml.py
/// straight from config.yaml, so the phone and the desktop pipeline can never drift
/// apart silently. The literals below are only the last-resort fallback.
struct CountingConfig: Codable {
    var tileSize: Int = 640
    var overlap: Double = 0.20
    var edgeMarginRatio: Double = 0.10
    var iouThreshold: Double = 0.35
    var scoreThreshold: Double = 0.25
    var maxDetectionsPerTile: Int = 1200
    var modelName: String = "RebarDetector"

    static let `default` = CountingConfig()

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
