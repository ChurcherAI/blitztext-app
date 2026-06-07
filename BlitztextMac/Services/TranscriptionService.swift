import Foundation

enum TranscriptionError: LocalizedError {
    case noFile
    case notConfigured(TranscriptionProvider)
    case networkError(String)
    case apiError(TranscriptionProvider, String)

    var errorDescription: String? {
        switch self {
        case .noFile:
            return "Keine Audio-Datei gefunden"
        case .notConfigured(let provider):
            return "\(provider.keychainKey.label) fehlt. Bitte in den Einstellungen hinterlegen."
        case .networkError(let msg):
            return "Netzwerkfehler: \(msg)"
        case .apiError(let provider, let msg):
            return "\(provider.displayName)-Fehler: \(msg)"
        }
    }
}

private struct TranscriptionTextResponse: Decodable {
    let text: String?
}

private struct TranscriptionAPIErrorResponse: Decodable {
    struct APIError: Decodable {
        let message: String?
    }

    let error: APIError?
    let message: String?
}

enum TranscriptionService {
    private static let openAIModel = "whisper-1"
    private static let openAITranscriptionsURL = URL(string: "https://api.openai.com/v1/audio/transcriptions")!

    private static let mistralModel = "voxtral-mini-latest"
    private static let mistralTranscriptionsURL = URL(string: "https://api.mistral.ai/v1/audio/transcriptions")!

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 60
        return URLSession(configuration: configuration)
    }()

    static func transcribe(
        audioURL: URL,
        provider: TranscriptionProvider = .openAIWhisper,
        customTerms: [String] = [],
        language: String? = nil
    ) async throws -> String {
        guard let apiKey = KeychainService.load(key: provider.keychainKey) else {
            throw TranscriptionError.notConfigured(provider)
        }

        let endpointURL: URL
        let model: String
        switch provider {
        case .openAIWhisper:
            endpointURL = openAITranscriptionsURL
            model = openAIModel
        case .mistralVoxtral:
            endpointURL = mistralTranscriptionsURL
            model = mistralModel
        }

        return try await Task.detached(priority: .userInitiated) {
            defer {
                try? FileManager.default.removeItem(at: audioURL)
            }

            let boundary = UUID().uuidString
            var request = URLRequest(url: endpointURL)
            request.httpMethod = "POST"
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
            request.setValue("text/plain, application/json", forHTTPHeaderField: "Accept")
            request.timeoutInterval = 60
            request.cachePolicy = .reloadIgnoringLocalCacheData

            let audioData = try Data(contentsOf: audioURL, options: [.mappedIfSafe])

            var body = Data()
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.m4a\"\r\n")
            body.append("Content-Type: audio/m4a\r\n\r\n")
            body.append(audioData)
            body.append("\r\n")

            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
            body.append(model)
            body.append("\r\n")

            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n")
            body.append("text")
            body.append("\r\n")

            if !customTerms.isEmpty {
                let prompt = "Eigennamen und Begriffe: \(customTerms.joined(separator: ", "))"
                body.append("--\(boundary)\r\n")
                body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n")
                body.append(prompt)
                body.append("\r\n")
            }

            if let language, !language.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                body.append("--\(boundary)\r\n")
                body.append("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
                body.append(language.trimmingCharacters(in: .whitespacesAndNewlines))
                body.append("\r\n")
            }

            body.append("--\(boundary)--\r\n")
            request.httpBody = body

            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw TranscriptionError.networkError("Ungueltige Antwort")
            }

            guard httpResponse.statusCode == 200 else {
                throw TranscriptionError.apiError(provider, errorMessage(from: data) ?? "Status \(httpResponse.statusCode)")
            }

            guard let rawText = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                  !rawText.isEmpty else {
                throw TranscriptionError.apiError(provider, "Transkription fehlgeschlagen")
            }

            let text = extractedText(from: data, fallback: rawText)

            guard !text.isEmpty else {
                throw TranscriptionError.apiError(provider, "Transkription fehlgeschlagen")
            }

            return text
        }.value
    }

    /// Some providers (e.g. Mistral) return a JSON object even when `response_format=text`
    /// is requested. Unwrap the `text` field in that case; otherwise use the raw response.
    private static func extractedText(from data: Data, fallback: String) -> String {
        guard fallback.hasPrefix("{"),
              let decoded = try? JSONDecoder().decode(TranscriptionTextResponse.self, from: data),
              let text = decoded.text?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            return fallback
        }
        return text
    }

    private static func errorMessage(from data: Data) -> String? {
        guard let decoded = try? JSONDecoder().decode(TranscriptionAPIErrorResponse.self, from: data) else {
            return nil
        }
        return decoded.error?.message ?? decoded.message
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
