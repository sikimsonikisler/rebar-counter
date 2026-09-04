import SwiftData
import SwiftUI

struct RecordListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \CountRecord.createdAt, order: .reverse) private var records: [CountRecord]

    var body: some View {
        Group {
            if records.isEmpty {
                ContentUnavailableView(
                    "Henüz sayım yok",
                    systemImage: "circle.grid.3x3",
                    description: Text("İstifin ucunun fotoğrafını çekerek başla.")
                )
            } else {
                List {
                    ForEach(records) { record in
                        NavigationLink(value: record.id) {
                            RecordRow(record: record)
                        }
                    }
                    .onDelete(perform: delete)
                }
                .navigationDestination(for: PersistentIdentifier.self) { id in
                    if let record = records.first(where: { $0.id == id }) {
                        RecordDetailView(record: record)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        NavigationLink {
                            FieldTestSummaryView(records: records)
                        } label: {
                            Image(systemName: "chart.bar.doc.horizontal")
                        }
                    }
                }
            }
        }
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            let record = records[index]
            // The photo lives outside the database, so it has to go too.
            PhotoStore.delete(record.photoFileName)
            modelContext.delete(record)
        }
        try? modelContext.save()
    }
}

private struct RecordRow: View {
    let record: CountRecord

    var body: some View {
        HStack(spacing: 12) {
            Text("\(record.count)")
                .font(.system(.title2, design: .rounded, weight: .bold))
                .monospacedDigit()
                .frame(minWidth: 56, alignment: .trailing)

            VStack(alignment: .leading, spacing: 2) {
                Text(record.deliveryNote.isEmpty ? "İrsaliyesiz" : record.deliveryNote)
                    .font(.body)
                    .foregroundStyle(record.deliveryNote.isEmpty ? .secondary : .primary)
                Text("Ø\(record.diameterMM) · \(record.createdAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if record.correctionCount > 0 {
                Text("±\(record.correctionCount)")
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(.tertiarySystemFill), in: Capsule())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
