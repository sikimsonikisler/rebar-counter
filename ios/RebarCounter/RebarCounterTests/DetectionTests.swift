import Testing
import CoreGraphics
@testable import RebarCounter

/// The editing rules behind SPEC 3 step 4 - the part of the app that has to be right
/// even when the model is not.
struct DetectionTests {
    private let bars = [
        Detection(x: 100, y: 100, radius: 20),
        Detection(x: 160, y: 100, radius: 20),
        Detection(x: 100, y: 160, radius: 22)
    ]

    @Test("Tapping a marked bar finds that bar")
    func hitTestFindsTheBarUnderTheFinger() {
        let hit = bars.hitTest(CGPoint(x: 104, y: 96), tolerance: 10)
        #expect(hit?.x == 100)
        #expect(hit?.y == 100)
    }

    @Test("Tapping between bars finds nothing, so a new dot is added instead")
    func hitTestMissesTheGap() {
        #expect(bars.hitTest(CGPoint(x: 130, y: 160), tolerance: 4) == nil)
    }

    @Test("When two bars overlap the finger, the nearer one wins")
    func hitTestPrefersTheNearest() {
        let hit = bars.hitTest(CGPoint(x: 145, y: 100), tolerance: 30)
        #expect(hit?.x == 160)
    }

    @Test("A hand-added dot gets the typical bar size of this photo")
    func medianRadiusDescribesTheStack() {
        #expect(bars.medianRadius == 20)
        #expect([Detection]().medianRadius == 12)  // sensible default on an empty photo
    }

    @Test("Points survive a round trip through the record's stored blob")
    func pointsEncodeAndDecode() throws {
        let record = CountRecord(
            deliveryNote: "A-1", diameterMM: 12, note: "", count: 3, automaticCount: 3,
            correctionCount: 0, photoFileName: "x.jpg", points: bars, elapsedSeconds: 12
        )
        #expect(record.points.count == 3)
        #expect(record.points[2].radius == 22)
    }

    @Test("Difference against a hand count is reported only when there is one")
    func manualDifference() {
        let withManual = CountRecord(
            deliveryNote: "", diameterMM: 12, note: "", count: 418, automaticCount: 415,
            correctionCount: 3, photoFileName: "", points: [], elapsedSeconds: 20,
            manualCount: 420, manualSeconds: 300
        )
        #expect(withManual.manualDifference == -2)

        let withoutManual = CountRecord(
            deliveryNote: "", diameterMM: 12, note: "", count: 418, automaticCount: 418,
            correctionCount: 0, photoFileName: "", points: [], elapsedSeconds: 20
        )
        #expect(withoutManual.manualDifference == nil)
    }
}
