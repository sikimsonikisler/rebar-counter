import AVFoundation
import SwiftUI

/// Single-frame capture with a guide frame. SPEC 3 step 1-2: one photo, no video.
struct CameraView: View {
    var onCapture: (UIImage) -> Void
    var onCancel: () -> Void

    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if camera.isAuthorised {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
                GuideFrame()
            } else {
                PermissionNotice(state: camera.authorisationState) { camera.requestAccess() }
            }

            VStack {
                HStack {
                    Button("Vazgeç", action: onCancel)
                        .padding()
                        .foregroundStyle(.white)
                    Spacer()
                }
                Spacer()
                Text("İstifin ucu çerçeveye sığsın, mümkün olduğunca dik çek")
                    .font(.callout)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.bottom, 12)
                    .shadow(radius: 4)

                Button {
                    camera.capture()
                } label: {
                    ZStack {
                        Circle().fill(.white).frame(width: 74, height: 74)
                        Circle().stroke(.white.opacity(0.6), lineWidth: 4).frame(width: 88, height: 88)
                    }
                }
                .disabled(!camera.isAuthorised || camera.isCapturing)
                .padding(.bottom, 28)
            }
        }
        .task { await camera.start() }
        .onDisappear { camera.stop() }
        .onChange(of: camera.capturedImage) { _, image in
            if let image { onCapture(image) }
        }
    }
}

/// The rule of thumb that decides whether a photo is countable at all, drawn on screen.
private struct GuideFrame: View {
    var body: some View {
        GeometryReader { geometry in
            let inset = geometry.size.width * 0.06
            let rect = CGRect(
                x: inset,
                y: geometry.size.height * 0.18,
                width: geometry.size.width - inset * 2,
                height: geometry.size.height * 0.52
            )
            Path { path in
                path.addRoundedRect(in: rect, cornerSize: CGSize(width: 14, height: 14))
            }
            .stroke(.white.opacity(0.85), style: StrokeStyle(lineWidth: 2, dash: [10, 8]))
        }
        .allowsHitTesting(false)
    }
}

private struct PermissionNotice: View {
    var state: AVAuthorizationStatus
    var onRequest: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill").font(.system(size: 44)).foregroundStyle(.white)
            Text(message).foregroundStyle(.white).multilineTextAlignment(.center)
            if state == .notDetermined {
                Button("Kamerayı Aç", action: onRequest).buttonStyle(.borderedProminent)
            } else if state == .denied || state == .restricted {
                Button("Ayarları Aç") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(32)
    }

    private var message: String {
        switch state {
        case .denied, .restricted: "Kamera izni kapalı. Ayarlardan açabilirsin."
        default: "Sayım için kamera izni gerekiyor."
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

@MainActor
final class CameraController: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var isCapturing = false
    @Published var authorisationState: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)

    let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private var isConfigured = false

    var isAuthorised: Bool { authorisationState == .authorized }

    func requestAccess() {
        Task {
            _ = await AVCaptureDevice.requestAccess(for: .video)
            authorisationState = AVCaptureDevice.authorizationStatus(for: .video)
            await start()
        }
    }

    func start() async {
        authorisationState = AVCaptureDevice.authorizationStatus(for: .video)
        guard isAuthorised else { return }
        if !isConfigured { configure() }
        guard !session.isRunning else { return }
        let session = self.session
        await Task.detached { session.startRunning() }.value
    }

    func stop() {
        guard session.isRunning else { return }
        let session = self.session
        Task.detached { session.stopRunning() }
    }

    private func configure() {
        session.beginConfiguration()
        // Full resolution: a bar end is only a few dozen pixels wide, and every pixel
        // thrown away here is a bar the model may miss.
        session.sessionPreset = .photo

        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        session.commitConfiguration()
        isConfigured = true
    }

    func capture() {
        guard isAuthorised, !isCapturing else { return }
        isCapturing = true
        let settings = AVCapturePhotoSettings()
        settings.flashMode = .auto
        output.capturePhoto(with: settings, delegate: self)
    }
}

extension CameraController: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        let image = photo.fileDataRepresentation().flatMap(UIImage.init(data:))
        Task { @MainActor in
            self.isCapturing = false
            if let image { self.capturedImage = image.normalisedUp() }
        }
    }
}

extension UIImage {
    /// Counting works in pixel coordinates, so bake the EXIF rotation into the pixels
    /// once instead of carrying an orientation flag through the whole pipeline.
    func normalisedUp() -> UIImage {
        guard imageOrientation != .up else { return self }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
