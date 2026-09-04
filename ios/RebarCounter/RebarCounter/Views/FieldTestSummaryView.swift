import SwiftData
import SwiftUI
import UniformTypeIdentifiers

/// SPEC 6 phase 5, measured on the device that did the counting.
///
/// The acceptance criteria are an average time under 30 s and fewer mistakes than a
/// hand count, so both numbers are computed here from the records themselves rather
/// than from anybody's recollection. Export is a share sheet the user triggers - the
/// data never leaves the phone on its own.
struct FieldTestSummaryView: View {
    let records: [CountRecord]

    private var compared: [CountRecord] {
        records.filter { $0.manualCount != nil }
    }

    var body: some View {
        List {
            Section("Uygulama") {
                metric("Sayım adedi", "\(records.count)")
                metric("Ortalama süre", seconds(averageElapsed))
                metric("Ortalama düzeltme", String(format: "%.1f dokunuş", averageCorrections))
            }

            if compared.isEmpty {
                Section {
                    Text("Elle sayım girilen kayıt yok. Kaydet ekranındaki “Saha testi” bölümünü doldurunca karşılaştırma burada çıkar.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            } else {
                Section("Elle sayımla karşılaştırma (\(compared.count) kayıt)") {
                    metric("Uygulama ort. süre", seconds(averageElapsed(of: compared)))
                    metric("Elle ort. süre", seconds(averageManualSeconds))
                    metric("Ortalama fark", String(format: "%+.1f adet", averageDifference))
                    metric("Farkın büyüklüğü", String(format: "%.1f adet", averageAbsoluteDifference))
                }
                Section {
                    verdict
                }
            }

            Section {
                ShareLink(
                    item: csv,
                    preview: SharePreview("saha-testi.csv")
                ) {
                    Label("CSV olarak dışa aktar", systemImage: "square.and.arrow.up")
                }
            } footer: {
                Text("Dosya yalnızca sen paylaşınca çıkar; uygulama hiçbir şey göndermez.")
            }
        }
        .navigationTitle("Saha Testi")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Numbers

    private var averageElapsed: Double { averageElapsed(of: records) }

    private func averageElapsed(of list: [CountRecord]) -> Double {
        guard !list.isEmpty else { return 0 }
        return list.reduce(0) { $0 + $1.elapsedSeconds } / Double(list.count)
    }

    private var averageCorrections: Double {
        guard !records.isEmpty else { return 0 }
        return records.reduce(0) { $0 + Double($1.correctionCount) } / Double(records.count)
    }

    private var averageManualSeconds: Double {
        let values = compared.compactMap(\.manualSeconds)
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private var averageDifference: Double {
        let values = compared.compactMap(\.manualDifference).map(Double.init)
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private var averageAbsoluteDifference: Double {
        let values = compared.compactMap(\.manualDifference).map { Double(abs($0)) }
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    @ViewBuilder
    private var verdict: some View {
        let fastEnough = averageElapsed(of: compared) < 30
        Label(
            fastEnough
                ? "Ortalama süre 30 sn hedefinin altında."
                : "Ortalama süre 30 sn hedefinin üstünde.",
            systemImage: fastEnough ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
        )
        .foregroundStyle(fastEnough ? .green : .orange)
        .font(.footnote)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        LabeledContent(label) { Text(value).monospacedDigit() }
    }

    private func seconds(_ value: Double) -> String {
        String(format: "%.0f sn", value)
    }

    // MARK: - Export

    private var csv: String {
        var lines = [
            "tarih;irsaliye;cap;otomatik;duzeltilmis;duzeltme;sure_sn;elle_sayim;elle_sure_sn;fark"
        ]
        let formatter = ISO8601DateFormatter()
        for record in records.sorted(by: { $0.createdAt < $1.createdAt }) {
            lines.append(
                [
                    formatter.string(from: record.createdAt),
                    record.deliveryNote,
                    "\(record.diameterMM)",
                    "\(record.automaticCount)",
                    "\(record.count)",
                    "\(record.correctionCount)",
                    String(format: "%.1f", record.elapsedSeconds),
                    record.manualCount.map(String.init) ?? "",
                    record.manualSeconds.map { String(format: "%.1f", $0) } ?? "",
                    record.manualDifference.map { String($0) } ?? ""
                ].joined(separator: ";")
            )
        }
        return lines.joined(separator: "\n")
    }
}
