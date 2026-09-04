import UIKit

/// SPEC 3 step 6: the marked photo, the number and the note as ONE image.
///
/// One image, because it goes to WhatsApp: a picture survives forwarding, a
/// structured file does not.
enum ShareRenderer {
    static func render(
        image: UIImage,
        detections: [Detection],
        count: Int,
        deliveryNote: String,
        diameterMM: Int,
        note: String,
        date: Date
    ) -> UIImage {
        let width = image.size.width
        let headerHeight = max(96, width * 0.10)
        let footerHeight = note.isEmpty ? 0 : max(72, width * 0.075)
        let canvasSize = CGSize(width: width, height: image.size.height + headerHeight + footerHeight)

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: canvasSize, format: format).image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: canvasSize))

            image.draw(in: CGRect(x: 0, y: headerHeight, width: width, height: image.size.height))
            drawMarkers(detections, in: context.cgContext, offsetY: headerHeight)

            drawHeader(
                count: count,
                deliveryNote: deliveryNote,
                diameterMM: diameterMM,
                date: date,
                in: CGRect(x: 0, y: 0, width: width, height: headerHeight)
            )
            if !note.isEmpty {
                drawNote(
                    note,
                    in: CGRect(
                        x: 0, y: headerHeight + image.size.height,
                        width: width, height: footerHeight
                    )
                )
            }
        }
    }

    private static func drawMarkers(_ detections: [Detection], in context: CGContext, offsetY: CGFloat) {
        context.saveGState()
        context.setLineWidth(max(1, detections.first.map { $0.radius * 0.08 } ?? 1))
        for detection in detections {
            let radius = detection.radius
            let centre = CGPoint(x: detection.x, y: detection.y + offsetY)

            context.setStrokeColor(UIColor.systemGreen.withAlphaComponent(0.9).cgColor)
            context.strokeEllipse(
                in: CGRect(x: centre.x - radius, y: centre.y - radius, width: radius * 2, height: radius * 2)
            )
            let dot = max(1.5, radius * 0.32)
            context.setFillColor(UIColor.systemRed.cgColor)
            context.fillEllipse(
                in: CGRect(x: centre.x - dot, y: centre.y - dot, width: dot * 2, height: dot * 2)
            )
        }
        context.restoreGState()
    }

    private static func drawHeader(
        count: Int,
        deliveryNote: String,
        diameterMM: Int,
        date: Date,
        in rect: CGRect
    ) {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd.MM.yyyy HH:mm"

        var subtitle = "Ø\(diameterMM)"
        if !deliveryNote.isEmpty { subtitle += " · İrsaliye \(deliveryNote)" }
        subtitle += " · \(formatter.string(from: date))"

        let titleFont = UIFont.systemFont(ofSize: rect.height * 0.44, weight: .bold)
        let subtitleFont = UIFont.systemFont(ofSize: rect.height * 0.20, weight: .regular)

        ("\(count) adet" as NSString).draw(
            at: CGPoint(x: rect.width * 0.04, y: rect.height * 0.10),
            withAttributes: [.font: titleFont, .foregroundColor: UIColor.white]
        )
        (subtitle as NSString).draw(
            at: CGPoint(x: rect.width * 0.04, y: rect.height * 0.62),
            withAttributes: [.font: subtitleFont, .foregroundColor: UIColor.lightGray]
        )
    }

    private static func drawNote(_ note: String, in rect: CGRect) {
        let font = UIFont.systemFont(ofSize: rect.height * 0.30, weight: .regular)
        let inset = rect.insetBy(dx: rect.width * 0.04, dy: rect.height * 0.20)
        (note as NSString).draw(
            with: inset,
            options: [.usesLineFragmentOrigin],
            attributes: [.font: font, .foregroundColor: UIColor.white],
            context: nil
        )
    }
}
