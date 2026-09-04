import CoreGraphics
import Foundation

/// Overlapping tiling and merge-back - the Swift twin of python/ml/tile.py.
///
/// A photo holds 300-600 cross-sections. Squeezed into one 640x640 model input each
/// one would be a few pixels wide and invisible, so the image is cut into overlapping
/// tiles and the results are merged back. Kept deliberately parallel to the Python
/// version, function for function, so the two can be compared when they disagree.
enum Tiler {
    struct Tile {
        var rect: CGRect
        var touchesLeft: Bool
        var touchesTop: Bool
        var touchesRight: Bool
        var touchesBottom: Bool
    }

    struct Box {
        var rect: CGRect
        var score: Double
    }

    /// Tile origins along one axis; the last one is pulled back to end on the edge,
    /// so every tile is exactly `tileSize` wide unless the image itself is smaller.
    static func starts(length: Int, tileSize: Int, step: Int) -> [Int] {
        guard length > tileSize else { return [0] }
        var positions = Array(stride(from: 0, through: length - tileSize, by: step))
        if let last = positions.last, last + tileSize < length {
            positions.append(length - tileSize)
        }
        if positions.isEmpty { positions = [length - tileSize] }
        return Array(Set(positions)).sorted()
    }

    static func plan(width: Int, height: Int, config: CountingConfig) -> [Tile] {
        let tileSize = config.tileSize
        let step = max(1, Int((Double(tileSize) * (1.0 - config.overlap)).rounded()))

        var tiles: [Tile] = []
        for top in starts(length: height, tileSize: tileSize, step: step) {
            for left in starts(length: width, tileSize: tileSize, step: step) {
                let right = min(left + tileSize, width)
                let bottom = min(top + tileSize, height)
                tiles.append(
                    Tile(
                        rect: CGRect(x: left, y: top, width: right - left, height: bottom - top),
                        touchesLeft: left == 0,
                        touchesTop: top == 0,
                        touchesRight: right >= width,
                        touchesBottom: bottom >= height
                    )
                )
            }
        }
        return tiles
    }

    /// Boxes hugging an *interior* tile edge are cut in half by the crop, so they are
    /// dropped - the neighbouring tile sees the same bar whole. A tile edge that is
    /// also the image edge is real and kept.
    static func keep(box: CGRect, in tile: Tile, margin: Double) -> Bool {
        let cx = box.midX
        let cy = box.midY
        if !tile.touchesLeft, cx < margin { return false }
        if !tile.touchesTop, cy < margin { return false }
        if !tile.touchesRight, cx > tile.rect.width - margin { return false }
        if !tile.touchesBottom, cy > tile.rect.height - margin { return false }
        return true
    }

    /// Must not exceed half the overlap, or a bar could be dropped by both tiles that
    /// see it. A duplicate is free (NMS eats it); a loss is a wrong count.
    static func edgeMargin(_ config: CountingConfig) -> Double {
        let ratio = min(config.edgeMarginRatio, config.overlap / 2.0)
        return ratio * Double(config.tileSize)
    }

    static func toGlobal(_ box: CGRect, tile: Tile) -> CGRect {
        box.offsetBy(dx: tile.rect.minX, dy: tile.rect.minY)
    }

    /// Greedy non-maximum suppression, bucketed by a uniform grid.
    ///
    /// A whole-photo merge sees several thousand boxes, and comparing every pair would
    /// be tens of millions of rectangle intersections on a phone. Cross-sections are
    /// all roughly one size, so hashing them into cells the size of a bar means each
    /// candidate only ever meets its actual neighbours. The result is identical to the
    /// naive version - same greedy order, same threshold.
    static func nms(_ boxes: [Box], iouThreshold: Double) -> [Box] {
        guard boxes.count > 1 else { return boxes }

        let order = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
        // Cell as wide as a typical box: two boxes can only overlap if they share a
        // cell or sit in adjacent ones.
        let typicalSide = boxes.reduce(0.0) { $0 + Double($1.rect.width + $1.rect.height) }
            / Double(boxes.count * 2)
        let cell = Swift.max(typicalSide, 1.0)

        var grid: [Cell: [Int]] = [:]
        grid.reserveCapacity(boxes.count)
        var suppressed = [Bool](repeating: false, count: boxes.count)
        var kept: [Box] = []
        kept.reserveCapacity(boxes.count)

        for index in order {
            if suppressed[index] { continue }
            let candidate = boxes[index]
            let home = Cell(candidate.rect.midX, candidate.rect.midY, cell)

            var overlapsKept = false
            for neighbour in home.neighbourhood {
                guard let bucket = grid[neighbour] else { continue }
                for other in bucket where iou(candidate.rect, boxes[other].rect) > iouThreshold {
                    overlapsKept = true
                    break
                }
                if overlapsKept { break }
            }
            if overlapsKept {
                suppressed[index] = true
                continue
            }

            kept.append(candidate)
            grid[home, default: []].append(index)
        }
        return kept
    }

    /// Integer grid coordinate used to bucket boxes for NMS.
    private struct Cell: Hashable {
        let x: Int
        let y: Int

        init(x: Int, y: Int) {
            self.x = x
            self.y = y
        }

        init(_ px: CGFloat, _ py: CGFloat, _ size: Double) {
            self.x = Int(floor(Double(px) / size))
            self.y = Int(floor(Double(py) / size))
        }

        var neighbourhood: [Cell] {
            var cells: [Cell] = []
            cells.reserveCapacity(9)
            for dy in -1...1 {
                for dx in -1...1 {
                    cells.append(Cell(x: x + dx, y: y + dy))
                }
            }
            return cells
        }
    }

    static func iou(_ a: CGRect, _ b: CGRect) -> Double {
        let intersection = a.intersection(b)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let intersectionArea = Double(intersection.width * intersection.height)
        let union = Double(a.width * a.height) + Double(b.width * b.height) - intersectionArea
        return union > 0 ? intersectionArea / union : 0
    }

    static func detections(from boxes: [Box]) -> [Detection] {
        boxes.map { box in
            Detection(
                x: Double(box.rect.midX),
                y: Double(box.rect.midY),
                radius: Double(box.rect.width + box.rect.height) / 4.0
            )
        }
    }
}
