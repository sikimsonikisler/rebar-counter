import Foundation
import Testing
import CoreGraphics
@testable import RebarCounter

/// Tiler is the Swift twin of python/ml/tile.py. If the two ever disagree, the phone
/// and the desktop pipeline count differently on the same photo - so the numbers below
/// are the ones the Python module actually produces, not the ones that look right.
struct TilerTests {
    let config = CountingConfig.default  // 640 px tiles, 20% overlap

    @Test("Tile plan matches python/ml/tile.py exactly")
    func planMatchesPython() {
        #expect(Tiler.plan(width: 1800, height: 1300, config: config).count == 12)
        #expect(Tiler.plan(width: 4032, height: 3024, config: config).count == 48)
        #expect(Tiler.plan(width: 500, height: 400, config: config).count == 1)
    }

    @Test("Last tile is pulled back onto the edge instead of hanging over it")
    func startsClampToEdge() {
        let starts = Tiler.starts(length: 1800, tileSize: 640, step: 512)
        #expect(starts == [0, 512, 1024, 1160])
        #expect(starts.last! + 640 == 1800)
    }

    @Test("Every pixel of the photo lands in at least one tile")
    func tilesCoverTheImage() {
        let width = 1800
        let height = 1300
        var covered = [Bool](repeating: false, count: width * height)
        for tile in Tiler.plan(width: width, height: height, config: config) {
            for y in Int(tile.rect.minY)..<Int(tile.rect.maxY) {
                for x in Int(tile.rect.minX)..<Int(tile.rect.maxX) {
                    covered[y * width + x] = true
                }
            }
        }
        #expect(!covered.contains(false))
    }

    @Test("A smaller-than-a-tile photo becomes exactly one tile")
    func smallImageIsOneTile() {
        let tiles = Tiler.plan(width: 300, height: 200, config: config)
        #expect(tiles.count == 1)
        #expect(tiles[0].rect == CGRect(x: 0, y: 0, width: 300, height: 200))
        #expect(tiles[0].touchesLeft && tiles[0].touchesRight)
    }

    @Test("Boxes on an interior tile edge are dropped, boxes on the image edge are kept")
    func edgeRule() {
        let margin = Tiler.edgeMargin(config)
        let interior = Tiler.Tile(
            rect: CGRect(x: 512, y: 512, width: 640, height: 640),
            touchesLeft: false, touchesTop: false, touchesRight: false, touchesBottom: false
        )
        let atLeftEdge = CGRect(x: 0, y: 300, width: 20, height: 20)
        #expect(!Tiler.keep(box: atLeftEdge, in: interior, margin: margin))

        let firstTile = Tiler.Tile(
            rect: CGRect(x: 0, y: 0, width: 640, height: 640),
            touchesLeft: true, touchesTop: true, touchesRight: false, touchesBottom: false
        )
        #expect(Tiler.keep(box: atLeftEdge, in: firstTile, margin: margin))
    }

    @Test("Edge margin can never exceed half the overlap")
    func marginIsClamped() {
        var greedy = CountingConfig.default
        greedy.overlap = 0.10
        greedy.edgeMarginRatio = 0.40  // would otherwise lose bars between tiles
        #expect(Tiler.edgeMargin(greedy) == 0.05 * Double(greedy.tileSize))
    }

    @Test("NMS removes a duplicate but keeps a genuinely different bar")
    func nmsCollapsesDuplicates() {
        let boxes = [
            Tiler.Box(rect: CGRect(x: 0, y: 0, width: 10, height: 10), score: 0.9),
            Tiler.Box(rect: CGRect(x: 1, y: 1, width: 10, height: 10), score: 0.8),
            Tiler.Box(rect: CGRect(x: 40, y: 0, width: 10, height: 10), score: 0.7)
        ]
        #expect(Tiler.nms(boxes, iouThreshold: 0.35).count == 2)
    }

    @Test("NMS keeps touching cross-sections apart")
    func nmsKeepsNeighbours() {
        // Bars in a stack touch; suppressing a neighbour would undercount every time.
        let boxes = [
            Tiler.Box(rect: CGRect(x: 0, y: 0, width: 20, height: 20), score: 0.9),
            Tiler.Box(rect: CGRect(x: 19, y: 0, width: 20, height: 20), score: 0.9)
        ]
        #expect(Tiler.nms(boxes, iouThreshold: 0.35).count == 2)
    }

    @Test("NMS survives a whole-photo merge without a quadratic blow-up")
    func nmsHandlesThousandsOfBoxes() {
        // A 4032x3024 photo of Ø12 bars is roughly this: ~4000 cross-sections, each
        // seen by two tiles because of the overlap, so ~8000 boxes reach the merge.
        var boxes: [Tiler.Box] = []
        for row in 0..<64 {
            for column in 0..<64 {
                let rect = CGRect(x: Double(column) * 50, y: Double(row) * 50, width: 46, height: 46)
                boxes.append(Tiler.Box(rect: rect, score: 0.9))
                // The duplicate the neighbouring tile reported for the same bar.
                boxes.append(Tiler.Box(rect: rect.offsetBy(dx: 1, dy: 1), score: 0.85))
            }
        }

        let started = Date()
        let kept = Tiler.nms(boxes, iouThreshold: 0.35)
        let seconds = Date().timeIntervalSince(started)

        #expect(kept.count == 64 * 64)  // every duplicate collapsed, no real bar lost
        #expect(seconds < 1.0)
    }

    @Test("Bucketed NMS agrees with the naive definition")
    func nmsMatchesTheNaiveVersion() {
        var generator = SystemRandomNumberGenerator()
        var boxes: [Tiler.Box] = []
        for _ in 0..<300 {
            let x = Double.random(in: 0..<800, using: &generator)
            let y = Double.random(in: 0..<800, using: &generator)
            let side = Double.random(in: 18..<26, using: &generator)
            boxes.append(
                Tiler.Box(
                    rect: CGRect(x: x, y: y, width: side, height: side),
                    score: Double.random(in: 0.3..<1.0, using: &generator)
                )
            )
        }

        // Naive greedy NMS, written out here as the reference definition.
        var expected: [Tiler.Box] = []
        for candidate in boxes.sorted(by: { $0.score > $1.score }) {
            if !expected.contains(where: { Tiler.iou($0.rect, candidate.rect) > 0.35 }) {
                expected.append(candidate)
            }
        }
        #expect(Tiler.nms(boxes, iouThreshold: 0.35).count == expected.count)
    }

    @Test("Tile-local boxes come back in image coordinates")
    func mapsBackToGlobal() {
        let tile = Tiler.Tile(
            rect: CGRect(x: 1024, y: 512, width: 640, height: 640),
            touchesLeft: false, touchesTop: false, touchesRight: false, touchesBottom: false
        )
        let global = Tiler.toGlobal(CGRect(x: 10, y: 20, width: 30, height: 30), tile: tile)
        #expect(global.origin == CGPoint(x: 1034, y: 532))
    }

    @Test("Boxes become the centre-and-radius the UI draws")
    func boxesBecomeDetections() {
        let detections = Tiler.detections(from: [
            Tiler.Box(rect: CGRect(x: 10, y: 20, width: 20, height: 20), score: 1)
        ])
        #expect(detections.count == 1)
        #expect(detections[0].x == 20)
        #expect(detections[0].y == 30)
        #expect(detections[0].radius == 10)
    }
}
