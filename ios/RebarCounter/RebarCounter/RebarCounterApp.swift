import SwiftData
import SwiftUI

@main
struct RebarCounterApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        // Local store only. SPEC 2: no server, no account, no sync - the database
        // never leaves the phone and neither does the photo.
        .modelContainer(for: CountRecord.self)
    }
}
