import PhotosUI
import SwiftData
import SwiftUI

struct RootView: View {
    @State private var flow: FlowRequest?
    @State private var pickedItem: PhotosPickerItem?
    @State private var modelWarning: String?

    /// Identifiable wrapper so `fullScreenCover(item:)` can drive the flow.
    private struct FlowRequest: Identifiable {
        let id = UUID()
        let source: CountingFlowView.Source
    }

    var body: some View {
        NavigationStack {
            RecordListView()
                .navigationTitle("Demir Sayım")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        PhotosPicker(selection: $pickedItem, matching: .images) {
                            Image(systemName: "photo.on.rectangle")
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) {
                    Button {
                        start(.camera)
                    } label: {
                        Label("Yeni Sayım", systemImage: "camera.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                    .background(.bar)
                }
        }
        .fullScreenCover(item: $flow) { request in
            CountingFlowView(source: request.source) { flow = nil }
        }
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    start(.image(image.normalisedUp()))
                }
                pickedItem = nil
            }
        }
        .alert("Model eksik", isPresented: .constant(modelWarning != nil)) {
            Button("Tamam") { modelWarning = nil }
        } message: {
            Text(modelWarning ?? "")
        }
    }

    private func start(_ source: CountingFlowView.Source) {
        guard CountingService.isModelAvailable() else {
            modelWarning = """
            Sayım modeli uygulamaya eklenmemiş.
            Repo kökünde `python python/ml/export_coreml.py` çalıştırıp \
            uygulamayı yeniden derle.
            """
            return
        }
        flow = FlowRequest(source: source)
    }
}
