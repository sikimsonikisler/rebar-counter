import CoreGraphics
import Foundation

/// One counted cross-section, in pixels of the original photo.
///
/// The whole product hangs on this being a *list of points* rather than a single
/// number: without points there is nothing for the user to correct, and correcting
/// is where the value is (SPEC 3).
struct Detection: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var x: Double
    var y: Double
    var radius: Double

    var point: CGPoint { CGPoint(x: x, y: y) }

    func distance(to other: CGPoint) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }
}

extension Array where Element == Detection {
    /// Typical bar size in this photo - used to size a dot the user adds by hand.
    var medianRadius: Double {
        guard !isEmpty else { return 12 }
        let sorted = map(\.radius).sorted()
        return sorted[sorted.count / 2]
    }

    /// The detection under the finger, if any. Generous by a small factor because a
    /// fingertip is bigger than a bar end.
    func hitTest(_ location: CGPoint, tolerance: Double) -> Detection? {
        var best: Detection?
        var bestDistance = Double.greatestFiniteMagnitude
        for detection in self {
            let distance = detection.distance(to: location)
            if distance <= Swift.max(detection.radius, tolerance), distance < bestDistance {
                best = detection
                bestDistance = distance
            }
        }
        return best
    }
}
