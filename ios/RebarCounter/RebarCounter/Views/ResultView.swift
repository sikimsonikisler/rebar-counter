import SwiftUI

/// SPEC 3 step 3-4, and the reason the whole product works.
///
/// 92% accuracy is fine if fixing the rest takes ten seconds, so this screen is not a
/// viewer with editing bolted on - correcting is the primary interaction. Pinch to
/// zoom, tap an unmarked bar to add it, tap a wrong dot to remove it, undo anything.
struct ResultView: View {
    let image: UIImage
    let automaticCount: Int
    var onSave: ([Detection], SaveDetails) -> Void
    var onRetake: () -> Void

    @State private var detections: [Detection]
    @State private var history: [[Detection]] = []
    @State private var scale: CGFloat = 1
    @State private var committedScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var committedOffset: CGSize = .zero
    @State private var showsSaveSheet = false

    private let maximumHistory = 60
    private let tapTargetPoints: CGFloat = 22

    init(
        image: UIImage,
        detections: [Detection],
        automaticCount: Int,
        onSave: @escaping ([Detection], SaveDetails) -> Void,
        onRetake: @escaping () -> Void
    ) {
        self.image = image
        self.automaticCount = automaticCount
        self.onSave = onSave
        self.onRetake = onRetake
        _detections = State(initialValue: detections)
    }

    var body: some View {
        VStack(spacing: 0) {
            counter
            canvas
            controls
        }
        .background(Color(.systemBackground))
        .sheet(isPresented: $showsSaveSheet) {
            SaveSheet(
                count: detections.count,
                automaticCount: automaticCount,
                onSave: { details in
                    showsSaveSheet = false
                    onSave(detections, details)
                }
            )
        }
    }

    // MARK: - Pieces

    private var counter: some View {
        VStack(spacing: 2) {
            Text("\(detections.count)")
                .font(.system(size: 52, weight: .bold, design: .rounded))
                .monospacedDigit()
                .contentTransition(.numericText())
                .animation(.snappy, value: detections.count)
            Text(correctionSummary)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
    }

    private var correctionSummary: String {
        let difference = detections.count - automaticCount
        if difference == 0 { return "adet — düzeltme yok" }
        return difference > 0 ? "adet — \(difference) eklendi" : "adet — \(-difference) silindi"
    }

    private var canvas: some View {
        GeometryReader { geometry in
            let fitted = fittedSize(in: geometry.size)
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .frame(width: fitted.width, height: fitted.height)
                DotLayer(detections: detections, imageSize: image.size, fittedSize: fitted)
                    .frame(width: fitted.width, height: fitted.height)
                    .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            // Both taps sit on the untransformed content, so a location arrives in the
            // image's own frame and needs one flat conversion, not an inverse matrix.
            // The double tap is declared first so a quick two-finger... two-tap reset
            // does not also add and remove a dot on the way through.
            .onTapGesture(count: 2) { _ in resetZoom() }
            .onTapGesture(count: 1) { location in handleTap(at: location, fitted: fitted) }
            .scaleEffect(scale)
            .offset(offset)
            .gesture(zoomAndPan(viewport: geometry.size, fitted: fitted))
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
        .background(Color(.systemGray6))
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                undo()
            } label: {
                Label("Geri Al", systemImage: "arrow.uturn.backward")
            }
            .disabled(history.isEmpty)

            Button {
                onRetake()
            } label: {
                Label("Yeniden Çek", systemImage: "camera")
            }

            Spacer()

            Button {
                showsSaveSheet = true
            } label: {
                Label("Kaydet", systemImage: "square.and.arrow.down")
                    .font(.headline)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(Color(.secondarySystemBackground))
    }

    // MARK: - Geometry

    private func fittedSize(in available: CGSize) -> CGSize {
        let ratio = min(available.width / image.size.width, available.height / image.size.height)
        return CGSize(width: image.size.width * ratio, height: image.size.height * ratio)
    }

    private func zoomAndPan(viewport: CGSize, fitted: CGSize) -> some Gesture {
        SimultaneousGesture(
            MagnifyGesture()
                .onChanged { value in
                    scale = min(max(committedScale * value.magnification, 1), 12)
                    offset = clamped(offset, viewport: viewport, fitted: fitted)
                }
                .onEnded { _ in
                    committedScale = scale
                    if scale <= 1.01 { resetZoom() } else { committedOffset = offset }
                },
            DragGesture()
                .onChanged { value in
                    // Panning an unzoomed photo would just slide it out of the frame.
                    guard scale > 1.01 else { return }
                    offset = clamped(
                        CGSize(
                            width: committedOffset.width + value.translation.width,
                            height: committedOffset.height + value.translation.height
                        ),
                        viewport: viewport,
                        fitted: fitted
                    )
                }
                .onEnded { _ in committedOffset = offset }
        )
    }

    /// Keep the photo covering the viewport: the user should never pan into grey.
    private func clamped(_ candidate: CGSize, viewport: CGSize, fitted: CGSize) -> CGSize {
        let scaled = CGSize(width: fitted.width * scale, height: fitted.height * scale)
        let slackX = max(0, (scaled.width - viewport.width) / 2)
        let slackY = max(0, (scaled.height - viewport.height) / 2)
        return CGSize(
            width: min(max(candidate.width, -slackX), slackX),
            height: min(max(candidate.height, -slackY), slackY)
        )
    }

    private func resetZoom() {
        withAnimation(.snappy) {
            scale = 1
            committedScale = 1
            offset = .zero
            committedOffset = .zero
        }
    }

    // MARK: - Editing

    private func handleTap(at location: CGPoint, fitted: CGSize) {
        guard fitted.width > 0 else { return }
        let fitScale = image.size.width / fitted.width
        let imagePoint = CGPoint(x: location.x * fitScale, y: location.y * fitScale)
        // A fingertip covers a fixed number of screen points; zoomed in, that is a
        // smaller area of the photo, which is exactly why zooming makes editing precise.
        let tolerance = Double(tapTargetPoints * fitScale / scale)

        pushHistory()
        if let existing = detections.hitTest(imagePoint, tolerance: tolerance) {
            detections.removeAll { $0.id == existing.id }
        } else {
            detections.append(
                Detection(x: imagePoint.x, y: imagePoint.y, radius: detections.medianRadius)
            )
        }
        UISelectionFeedbackGenerator().selectionChanged()
    }

    private func pushHistory() {
        history.append(detections)
        if history.count > maximumHistory { history.removeFirst() }
    }

    private func undo() {
        guard let previous = history.popLast() else { return }
        detections = previous
    }
}

/// Drawn in a Canvas rather than as hundreds of Views: a stack photo can carry 600
/// dots and SwiftUI will not keep up with 600 shape views while pinching.
struct DotLayer: View {
    let detections: [Detection]
    let imageSize: CGSize
    let fittedSize: CGSize

    var body: some View {
        Canvas { context, _ in
            guard imageSize.width > 0 else { return }
            let ratio = fittedSize.width / imageSize.width
            for detection in detections {
                let radius = max(1.5, detection.radius * ratio)
                let centre = CGPoint(x: detection.x * ratio, y: detection.y * ratio)

                let outline = Path(
                    ellipseIn: CGRect(
                        x: centre.x - radius, y: centre.y - radius,
                        width: radius * 2, height: radius * 2
                    )
                )
                context.stroke(outline, with: .color(.green.opacity(0.85)), lineWidth: 1)

                let dotRadius = max(1.5, radius * 0.32)
                let dot = Path(
                    ellipseIn: CGRect(
                        x: centre.x - dotRadius, y: centre.y - dotRadius,
                        width: dotRadius * 2, height: dotRadius * 2
                    )
                )
                context.fill(dot, with: .color(.red))
            }
        }
    }
}
