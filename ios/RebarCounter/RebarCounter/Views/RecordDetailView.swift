import SwiftData
import SwiftUI

struct RecordDetailView: View {
    let record: CountRecord

    @State private var shareImage: UIImage?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let image = PhotoStore.load(record.photoFileName) {
                    GeometryReader { geometry in
                        let fitted = fittedSize(image.size, in: geometry.size.width)
                        ZStack {
                            Image(uiImage: image).resizable().frame(width: fitted.width, height: fitted.height)
                            DotLayer(detections: record.points, imageSize: image.size, fittedSize: fitted)
                                .frame(width: fitted.width, height: fitted.height)
                        }
                    }
                    .aspectRatio(image.size, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    ContentUnavailableView("Fotoğraf bulunamadı", systemImage: "photo")
                        .frame(height: 180)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("\(record.count) adet")
                        .font(.system(size: 40, weight: .bold, design: .rounded))
                        .monospacedDigit()
                    detail("İrsaliye", record.deliveryNote.isEmpty ? "—" : record.deliveryNote)
                    detail("Çap", "Ø\(record.diameterMM)")
                    detail("Tarih", record.createdAt.formatted(date: .long, time: .shortened))
                    detail("Otomatik sayım", "\(record.automaticCount)")
                    detail("Elle düzeltme", "\(record.correctionCount)")
                    detail("Süre", String(format: "%.0f sn", record.elapsedSeconds))
                    if let manual = record.manualCount {
                        detail("Elle sayım", "\(manual)")
                        if let difference = record.manualDifference {
                            detail("Fark", difference == 0 ? "0" : String(format: "%+d", difference))
                        }
                        if let seconds = record.manualSeconds {
                            detail("Elle sayım süresi", String(format: "%.0f sn", seconds))
                        }
                    }
                    if !record.note.isEmpty {
                        Divider()
                        Text(record.note).font(.body)
                    }
                }
            }
            .padding()
        }
        .navigationTitle(record.deliveryNote.isEmpty ? "Sayım" : record.deliveryNote)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if let shareImage {
                    // One flattened image, because it is going to WhatsApp.
                    ShareLink(
                        item: Image(uiImage: shareImage),
                        preview: SharePreview("\(record.count) adet", image: Image(uiImage: shareImage))
                    ) {
                        Image(systemName: "square.and.arrow.up")
                    }
                } else {
                    ProgressView()
                }
            }
        }
        .task { shareImage = makeShareImage() }
    }

    private func detail(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.subheadline)
    }

    private func fittedSize(_ imageSize: CGSize, in width: CGFloat) -> CGSize {
        let ratio = width / imageSize.width
        return CGSize(width: width, height: imageSize.height * ratio)
    }

    private func makeShareImage() -> UIImage? {
        guard let image = PhotoStore.load(record.photoFileName) else { return nil }
        return ShareRenderer.render(
            image: image,
            detections: record.points,
            count: record.count,
            deliveryNote: record.deliveryNote,
            diameterMM: record.diameterMM,
            note: record.note,
            date: record.createdAt
        )
    }
}
