import Testing
import Foundation
@testable import RebarCounter

/// counting_config.json is written by python/ml/export_coreml.py from config.yaml.
/// These checks pin the contract between the two languages.
struct CountingConfigTests {
    @Test("Falls back to sane values when the bundle has no config")
    func fallbackMatchesConfigYaml() {
        let config = CountingConfig.loadFromBundle(Bundle(for: EmptyBundleMarker.self))
        #expect(config.tileSize == 640)
        #expect(config.overlap == 0.20)
        #expect(config.edgeMarginRatio == 0.10)
        #expect(config.modelName == "RebarDetector")
    }

    @Test("Reads the JSON the export script writes")
    func decodesExportedJSON() throws {
        let json = """
        {
          "tileSize": 512,
          "overlap": 0.25,
          "edgeMarginRatio": 0.10,
          "iouThreshold": 0.4,
          "scoreThreshold": 0.3,
          "maxDetectionsPerTile": 900,
          "modelName": "RebarDetector"
        }
        """
        let config = try JSONDecoder().decode(CountingConfig.self, from: Data(json.utf8))
        #expect(config.tileSize == 512)
        #expect(config.maxDetectionsPerTile == 900)
    }
}

private final class EmptyBundleMarker {}
