import Foundation
import SwiftData

/// A saved count. Metadata lives in SwiftData, the photo itself on disk (see PhotoStore):
/// a few hundred 3 MB photos inside the store would make every query drag.
@Model
final class CountRecord {
    var createdAt: Date = Date()
    /// Irsaliye numarası - the delivery note this count belongs to.
    var deliveryNote: String = ""
    var diameterMM: Int = 12
    var note: String = ""

    /// What the user finally accepted, after corrections.
    var count: Int = 0
    /// What the engine said before the user touched anything. Kept because the gap
    /// between the two is the honest measure of how good the model actually is.
    var automaticCount: Int = 0
    var correctionCount: Int = 0

    var photoFileName: String = ""
    var pointsData: Data = Data()

    /// SPEC 6 phase 5: shutter to save, measured by the app rather than a stopwatch.
    var elapsedSeconds: Double = 0
    /// Optional field-test comparison, entered by hand when someone also counts manually.
    var manualCount: Int?
    var manualSeconds: Double?

    init(
        deliveryNote: String,
        diameterMM: Int,
        note: String,
        count: Int,
        automaticCount: Int,
        correctionCount: Int,
        photoFileName: String,
        points: [Detection],
        elapsedSeconds: Double,
        manualCount: Int? = nil,
        manualSeconds: Double? = nil
    ) {
        self.createdAt = Date()
        self.deliveryNote = deliveryNote
        self.diameterMM = diameterMM
        self.note = note
        self.count = count
        self.automaticCount = automaticCount
        self.correctionCount = correctionCount
        self.photoFileName = photoFileName
        self.pointsData = (try? JSONEncoder().encode(points)) ?? Data()
        self.elapsedSeconds = elapsedSeconds
        self.manualCount = manualCount
        self.manualSeconds = manualSeconds
    }

    var points: [Detection] {
        (try? JSONDecoder().decode([Detection].self, from: pointsData)) ?? []
    }

    /// Difference against a hand count, when one was recorded.
    var manualDifference: Int? {
        guard let manualCount else { return nil }
        return count - manualCount
    }
}
