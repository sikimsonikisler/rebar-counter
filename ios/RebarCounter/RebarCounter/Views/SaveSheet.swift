import SwiftUI

/// What gets stored alongside the count (SPEC 3 step 5).
struct SaveDetails {
    var deliveryNote: String = ""
    var diameterMM: Int = 12
    var note: String = ""
    /// Field-test only (SPEC 6 phase 5); left empty in normal use.
    var manualCount: Int?
    var manualSeconds: Double?
}

struct SaveSheet: View {
    let count: Int
    let automaticCount: Int
    var onSave: (SaveDetails) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var details = SaveDetails()
    @State private var manualCountText = ""
    @State private var manualSecondsText = ""
    @State private var showsFieldTest = false

    // SPEC 7: the diameters that actually come off a truck.
    private let diameters = [8, 10, 12, 14, 16, 20]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Sayım") {
                        Text("\(count) adet").font(.headline).monospacedDigit()
                    }
                    if count != automaticCount {
                        LabeledContent("Otomatik") {
                            Text("\(automaticCount)").foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                }

                Section("İrsaliye") {
                    TextField("İrsaliye no", text: $details.deliveryNote)
                        .textInputAutocapitalization(.characters)
                    Picker("Çap", selection: $details.diameterMM) {
                        ForEach(diameters, id: \.self) { Text("Ø\($0)").tag($0) }
                    }
                    TextField("Not", text: $details.note, axis: .vertical)
                        .lineLimit(1...4)
                }

                Section {
                    DisclosureGroup("Saha testi", isExpanded: $showsFieldTest) {
                        TextField("Elle sayım (adet)", text: $manualCountText)
                            .keyboardType(.numberPad)
                        TextField("Elle sayım süresi (sn)", text: $manualSecondsText)
                            .keyboardType(.decimalPad)
                    }
                } footer: {
                    Text("Yalnızca saha testi sırasında doldurulur; normal kullanımda boş bırak.")
                }
            }
            .navigationTitle("Kaydet")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Vazgeç") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Kaydet") {
                        details.manualCount = Int(manualCountText)
                        details.manualSeconds = Double(manualSecondsText.replacingOccurrences(of: ",", with: "."))
                        onSave(details)
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}
