import UIKit

/// Photos on disk, metadata in SwiftData. Nothing here talks to the network.
enum PhotoStore {
    private static let folderName = "photos"

    private static var folder: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent(folderName, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func url(for fileName: String) -> URL {
        folder.appendingPathComponent(fileName)
    }

    /// Writes a JPEG and returns its file name. Quality is high on purpose: this photo
    /// is the evidence behind a delivery note, and it may be re-counted later.
    @discardableResult
    static func save(_ image: UIImage) throws -> String {
        let fileName = "\(UUID().uuidString).jpg"
        guard let data = image.jpegData(compressionQuality: 0.9) else {
            throw PhotoStoreError.encodingFailed
        }
        try data.write(to: url(for: fileName), options: .atomic)
        return fileName
    }

    static func load(_ fileName: String) -> UIImage? {
        guard !fileName.isEmpty else { return nil }
        return UIImage(contentsOfFile: url(for: fileName).path)
    }

    static func delete(_ fileName: String) {
        guard !fileName.isEmpty else { return }
        try? FileManager.default.removeItem(at: url(for: fileName))
    }
}

enum PhotoStoreError: LocalizedError {
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .encodingFailed: "Fotoğraf kaydedilemedi."
        }
    }
}
