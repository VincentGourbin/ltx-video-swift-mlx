// ModelDownloader.swift - HuggingFace Model Downloading for LTX-2
// Copyright 2025

import Foundation

// MARK: - Download Progress

/// Progress information for model downloads
public struct DownloadProgress: Sendable {
    /// Overall progress (0.0 to 1.0)
    public let progress: Double

    /// Current file being downloaded
    public let currentFile: String?

    /// Bytes downloaded so far
    public let bytesDownloaded: Int64

    /// Total bytes to download
    public let totalBytes: Int64

    /// Human-readable status message
    public let message: String

    public init(
        progress: Double,
        currentFile: String? = nil,
        bytesDownloaded: Int64 = 0,
        totalBytes: Int64 = 0,
        message: String = ""
    ) {
        self.progress = progress
        self.currentFile = currentFile
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
        self.message = message
    }
}

/// Callback type for download progress
public typealias DownloadProgressCallback = @Sendable (DownloadProgress) -> Void

// MARK: - Model Downloader

/// Downloads model weights from HuggingFace Hub
///
/// Uses per-component downloading:
/// - `vlm-gemma/` — Shared VLM Gemma 3 12B 4-bit QAT (text encoding + prompt enhancement)
/// - `connectors/` — Text encoder connector
/// - `vae/` — Video VAE decoder
/// - Unified safetensors file — Transformer weights (extracted from unified file)
public actor ModelDownloader {
    /// HuggingFace token for accessing gated models
    private let hfToken: String?

    /// Base cache directory
    internal let cacheDirectory: URL

    /// URLSession for downloads
    private let session: URLSession

    public init(hfToken: String? = nil, cacheDir: URL? = nil) {
        self.hfToken = hfToken ?? Self.resolveHFToken()

        // Use explicit cache directory, or fall back to LTXModelRegistry.modelsDirectory
        self.cacheDirectory = cacheDir ?? LTXModelRegistry.modelsDirectory

        // Create session with configuration
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600  // 1 hour timeout for large files
        self.session = URLSession(configuration: config)
    }

    /// Resolve an HF token from environment / on-disk credentials when one wasn't passed
    /// at construction time. Mirrors what `huggingface-cli login` writes:
    ///   1. `$HF_TOKEN` (preferred — explicit, scoped to the process)
    ///   2. `$HUGGING_FACE_HUB_TOKEN` (alternate name some tools use)
    ///   3. `~/.cache/huggingface/token` (written by `huggingface-cli login`)
    ///   4. `~/.huggingface/token` (legacy location)
    /// Returns nil if none are found — gated downloads will then fail explicitly.
    private static func resolveHFToken() -> String? {
        let env = ProcessInfo.processInfo.environment
        if let t = env["HF_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty {
            return t
        }
        if let t = env["HUGGING_FACE_HUB_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty {
            return t
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            home.appendingPathComponent(".cache/huggingface/token"),
            home.appendingPathComponent(".huggingface/token"),
        ]
        for url in candidates {
            if let data = try? Data(contentsOf: url),
               let raw = String(data: data, encoding: .utf8) {
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }

    // MARK: - HuggingFace API

    /// List files in a HuggingFace repository
    private func listRepoFiles(repoId: String) async throws -> [String] {
        let url = URL(string: "https://huggingface.co/api/models/\(repoId)")!

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw LTXError.downloadFailed("Failed to list repository files")
        }

        // Parse JSON response
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            throw LTXError.downloadFailed("Invalid repository response")
        }

        // Extract filenames
        let files = siblings.compactMap { $0["rfilename"] as? String }

        // Filter to relevant files
        return files.filter { file in
            file.hasSuffix(".safetensors") ||
            file.hasSuffix(".json") ||
            file == "tokenizer.model"
        }
    }

    /// Download a single file from HuggingFace
    private func downloadFile(
        repoId: String,
        filename: String,
        to destination: URL
    ) async throws {
        // Skip if file already exists
        if FileManager.default.fileExists(atPath: destination.path) {
            return
        }

        // Create parent directories
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw LTXError.downloadFailed("Failed to download \(filename)")
        }

        // Move to destination
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    // MARK: - Per-Component Downloads

    /// Cache subdirectory for a model variant
    private func componentCacheDir(model: LTXModel) -> URL {
        cacheDirectory.appendingPathComponent("ltx-\(model.rawValue)")
    }

    /// Download connector weights
    ///
    /// Downloads `connectors/diffusion_pytorch_model.safetensors`
    ///
    /// - Parameters:
    ///   - model: The LTX model variant
    ///   - progress: Optional progress callback
    /// - Returns: Path to the connector safetensors file
    public func downloadConnector(
        model: LTXModel = .distilled,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = model.huggingFaceRepo
        let localDir = componentCacheDir(model: model).appendingPathComponent("connectors")
        let destination = localDir.appendingPathComponent("diffusion_pytorch_model.safetensors")

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Connector weights already downloaded"))
            return destination
        }

        progress?(DownloadProgress(progress: 0.1, message: "Downloading connector weights..."))
        try await downloadFile(
            repoId: repoId,
            filename: "connectors/diffusion_pytorch_model.safetensors",
            to: destination
        )
        progress?(DownloadProgress(progress: 1.0, message: "Connector download complete"))
        return destination
    }

    /// Download VAE decoder weights
    ///
    /// Downloads `vae/diffusion_pytorch_model.safetensors`
    ///
    /// - Parameters:
    ///   - model: The LTX model variant
    ///   - progress: Optional progress callback
    /// - Returns: Path to the VAE safetensors file
    public func downloadVAE(
        model: LTXModel = .distilled,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = model.huggingFaceRepo
        let localDir = componentCacheDir(model: model).appendingPathComponent("vae")
        let destination = localDir.appendingPathComponent("diffusion_pytorch_model.safetensors")

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "VAE weights already downloaded"))
        } else {
            progress?(DownloadProgress(progress: 0.1, message: "Downloading VAE weights..."))
            try await downloadFile(
                repoId: repoId,
                filename: "vae/diffusion_pytorch_model.safetensors",
                to: destination
            )
            progress?(DownloadProgress(progress: 1.0, message: "VAE download complete"))
        }

        // Also download VAE config.json (small file, contains timestep_conditioning flag etc.)
        let configDest = localDir.appendingPathComponent("config.json")
        if !FileManager.default.fileExists(atPath: configDest.path) {
            try await downloadFile(
                repoId: repoId,
                filename: "vae/config.json",
                to: configDest
            )
        }

        return destination
    }

    // MARK: - Audio Model Downloads

    /// Download Audio VAE decoder weights
    ///
    /// Downloads `audio_vae/diffusion_pytorch_model.safetensors` (~100MB)
    public func downloadAudioVAE(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = "Lightricks/LTX-2"
        let localDir = cacheDirectory.appendingPathComponent("ltx-audio-vae")
        let destination = localDir.appendingPathComponent("diffusion_pytorch_model.safetensors")

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Audio VAE weights already downloaded"))
            return destination
        }

        try FileManager.default.createDirectory(at: localDir, withIntermediateDirectories: true)
        progress?(DownloadProgress(progress: 0.1, message: "Downloading audio VAE weights..."))
        try await downloadFile(
            repoId: repoId,
            filename: "audio_vae/diffusion_pytorch_model.safetensors",
            to: destination
        )
        progress?(DownloadProgress(progress: 1.0, message: "Audio VAE download complete"))
        return destination
    }

    /// Download Vocoder weights
    ///
    /// Downloads `vocoder/diffusion_pytorch_model.safetensors` (~106MB)
    public func downloadVocoder(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = "Lightricks/LTX-2"
        let localDir = cacheDirectory.appendingPathComponent("ltx-vocoder")
        let destination = localDir.appendingPathComponent("diffusion_pytorch_model.safetensors")

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Vocoder weights already downloaded"))
            return destination
        }

        try FileManager.default.createDirectory(at: localDir, withIntermediateDirectories: true)
        progress?(DownloadProgress(progress: 0.1, message: "Downloading vocoder weights..."))
        try await downloadFile(
            repoId: repoId,
            filename: "vocoder/diffusion_pytorch_model.safetensors",
            to: destination
        )
        progress?(DownloadProgress(progress: 1.0, message: "Vocoder download complete"))
        return destination
    }

    /// Download unified weights file (contains transformer + VAE + connector)
    ///
    /// Used for the transformer component which may not have standalone files.
    /// The caller should extract only the needed keys.
    ///
    /// - Parameters:
    ///   - model: The LTX model variant
    ///   - progress: Optional progress callback
    /// - Returns: Path to the downloaded safetensors file
    public func downloadUnifiedWeights(
        model: LTXModel,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = model.huggingFaceRepo
        let filename = model.unifiedWeightsFilename
        let localDir = componentCacheDir(model: model)
        let destination = localDir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Unified weights already downloaded"))
            return destination
        }

        try FileManager.default.createDirectory(at: localDir, withIntermediateDirectories: true)

        progress?(DownloadProgress(progress: 0.1, currentFile: filename, message: "Downloading \(filename)..."))
        try await downloadFile(repoId: repoId, filename: filename, to: destination)
        progress?(DownloadProgress(progress: 1.0, message: "Unified weights download complete"))
        return destination
    }

    // MARK: - VLM Gemma (Shared 4-bit Model)

    /// HuggingFace repo for the shared VLM Gemma model (4-bit QAT, ~7.5GB)
    private static let vlmGemmaRepoID = "mlx-community/gemma-3-12b-it-qat-4bit"

    /// Files to download from the VLM Gemma repo
    private static let vlmGemmaFiles = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    ]

    /// Shared cache directory for VLM Gemma (used by all model variants)
    internal var vlmGemmaCacheDir: URL {
        cacheDirectory.appendingPathComponent("vlm-gemma")
    }

    /// Download VLM Gemma model (shared 4-bit QAT, ~7.5GB)
    ///
    /// Downloads `mlx-community/gemma-3-12b-it-qat-4bit` which provides:
    /// - Language model weights for text encoding (hidden state extraction)
    /// - Tokenizer files
    /// - VLM config (with quantization info parsed by Gemma3Config)
    ///
    /// This replaces per-variant text_encoder + tokenizer downloads, saving ~40GB disk space.
    /// The same VLM weights are used for both dev and distilled variants.
    ///
    /// - Parameter progress: Optional progress callback
    /// - Returns: Path to the VLM Gemma directory (contains model + tokenizer files)
    public func downloadVLMGemma(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let localDir = vlmGemmaCacheDir

        // Quick check: if config.json exists, assume already downloaded
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent("config.json").path) {
            progress?(DownloadProgress(progress: 1.0, message: "VLM Gemma already downloaded"))
            return localDir
        }

        // Also check MLXVLM cache location (may have been downloaded by I2V enhancement)
        let mlxvlmCacheDir = cacheDirectory.appendingPathComponent(
            "mlx-community/gemma-3-12b-it-qat-4bit")
        if FileManager.default.fileExists(atPath: mlxvlmCacheDir.appendingPathComponent("config.json").path) {
            // Symlink to our cache dir to avoid re-downloading
            try FileManager.default.createDirectory(
                at: localDir.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(
                at: localDir,
                withDestinationURL: mlxvlmCacheDir)
            progress?(DownloadProgress(progress: 1.0, message: "VLM Gemma found in MLXVLM cache"))
            return localDir
        }

        try FileManager.default.createDirectory(at: localDir, withIntermediateDirectories: true)

        progress?(DownloadProgress(progress: 0.0, message: "Downloading VLM Gemma (4-bit, ~7.5GB)..."))

        let totalFiles = Self.vlmGemmaFiles.count
        for (i, file) in Self.vlmGemmaFiles.enumerated() {
            progress?(DownloadProgress(
                progress: Double(i) / Double(totalFiles),
                currentFile: file,
                message: "Downloading vlm-gemma/\(file)..."
            ))
            try await downloadFile(
                repoId: Self.vlmGemmaRepoID,
                filename: file,
                to: localDir.appendingPathComponent(file)
            )
        }

        progress?(DownloadProgress(progress: 1.0, message: "VLM Gemma download complete"))
        return localDir
    }

    /// Download Gemma text encoder — uses shared VLM Gemma (4-bit QAT)
    ///
    /// Returns the VLM Gemma directory which contains both model weights and tokenizer.
    /// The `model` parameter is ignored since all variants share the same Gemma weights.
    public func downloadGemma(
        model: LTXModel = .distilled,
        progress: DownloadProgressCallback? = nil
    ) async throws -> (modelDir: URL, tokenizerDir: URL) {
        let vlmDir = try await downloadVLMGemma(progress: progress)
        return (modelDir: vlmDir, tokenizerDir: vlmDir)
    }

    /// Check if VLM Gemma is downloaded (shared across all model variants)
    public func isGemmaDownloaded(model: LTXModel = .distilled) -> Bool {
        return FileManager.default.fileExists(
            atPath: vlmGemmaCacheDir.appendingPathComponent("config.json").path)
    }

    /// Get the Gemma model and tokenizer directories (downloads if needed)
    public func getGemmaPaths(
        model: LTXModel = .distilled,
        progress: DownloadProgressCallback? = nil
    ) async throws -> (modelDir: URL, tokenizerDir: URL) {
        return try await downloadGemma(model: model, progress: progress)
    }

    // MARK: - Download All Components

    /// Download all components needed for generation
    ///
    /// Downloads (if not already cached):
    /// 1. VLM Gemma (text encoder + tokenizer)
    /// 2. Unified weights (transformer + VAE + connector in one file)
    ///
    /// - Parameters:
    ///   - model: The model variant
    ///   - progress: Optional progress callback
    /// - Returns: Paths to all downloaded components
    public func downloadAllComponents(
        model: LTXModel,
        progress: DownloadProgressCallback? = nil
    ) async throws -> LTXComponentPaths {
        progress?(DownloadProgress(progress: 0.0, message: "Downloading \(model.displayName) components..."))

        let vlmDir = try await downloadVLMGemma { p in
            progress?(DownloadProgress(progress: p.progress * 0.4, currentFile: p.currentFile, message: p.message))
        }

        let unifiedPath = try await downloadUnifiedWeights(model: model) { p in
            progress?(DownloadProgress(progress: 0.4 + p.progress * 0.6, currentFile: p.currentFile, message: p.message))
        }

        progress?(DownloadProgress(progress: 1.0, message: "All components downloaded"))

        return LTXComponentPaths(
            textEncoderDir: vlmDir,
            tokenizerDir: vlmDir,
            unifiedWeightsPath: unifiedPath
        )
    }

    // MARK: - Upscaler & LoRA Downloads

    /// Spatial upscaler filename on HuggingFace (LTX-2.3 x2 upscaler)
    public static let spatialUpscalerFilename = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

    /// Distilled LoRA filename on HuggingFace
    public static let distilledLoRAFilename = "ltx-2.3-22b-distilled-lora-384.safetensors"

    /// Download spatial upscaler weights
    public func downloadUpscalerWeights(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = "Lightricks/LTX-2.3"
        let filename = Self.spatialUpscalerFilename

        let weightsDir = cacheDirectory.appendingPathComponent("ltx-upscaler")
        let destination = weightsDir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Upscaler weights already downloaded"))
            return destination
        }

        progress?(DownloadProgress(progress: 0.1, message: "Downloading spatial upscaler weights..."))
        try await downloadFile(repoId: repoId, filename: filename, to: destination)
        progress?(DownloadProgress(progress: 1.0, message: "Upscaler download complete"))
        return destination
    }

    /// Check if spatial upscaler weights are downloaded
    public func isUpscalerDownloaded() -> Bool {
        let destination = cacheDirectory.appendingPathComponent("ltx-upscaler/\(Self.spatialUpscalerFilename)")
        return FileManager.default.fileExists(atPath: destination.path)
    }

    /// Download distilled LoRA weights
    public func downloadDistilledLoRA(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = "Lightricks/LTX-2.3"
        let filename = Self.distilledLoRAFilename

        let weightsDir = cacheDirectory.appendingPathComponent("ltx-lora")
        let destination = weightsDir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "Distilled LoRA already downloaded"))
            return destination
        }

        progress?(DownloadProgress(progress: 0.1, message: "Downloading distilled LoRA weights..."))
        try await downloadFile(repoId: repoId, filename: filename, to: destination)
        progress?(DownloadProgress(progress: 1.0, message: "Distilled LoRA download complete"))
        return destination
    }

    /// Check if distilled LoRA weights are downloaded
    public func isDistilledLoRADownloaded() -> Bool {
        let destination = cacheDirectory.appendingPathComponent("ltx-lora/\(Self.distilledLoRAFilename)")
        return FileManager.default.fileExists(atPath: destination.path)
    }

    /// LipDub IC-LoRA filename on HuggingFace
    public static let lipDubLoRAFilename = "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"

    /// Download the LipDub IC-LoRA from `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub`.
    /// Repo is gated; the user must have accepted the license on HF and provided
    /// `hfToken` at downloader construction time.
    public func downloadLipDubLoRA(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub"
        let filename = Self.lipDubLoRAFilename

        let weightsDir = cacheDirectory.appendingPathComponent("ltx-lora-lipdub")
        let destination = weightsDir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(progress: 1.0, message: "LipDub IC-LoRA already downloaded"))
            return destination
        }

        progress?(DownloadProgress(progress: 0.1, message: "Downloading LipDub IC-LoRA weights..."))
        try await downloadFile(repoId: repoId, filename: filename, to: destination)
        progress?(DownloadProgress(progress: 1.0, message: "LipDub IC-LoRA download complete"))
        return destination
    }

    /// Check if LipDub IC-LoRA weights are downloaded
    public func isLipDubLoRADownloaded() -> Bool {
        let destination = cacheDirectory.appendingPathComponent("ltx-lora-lipdub/\(Self.lipDubLoRAFilename)")
        return FileManager.default.fileExists(atPath: destination.path)
    }

    /// Clear downloaded models
    public func clearCache() throws {
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
        }
    }

    /// Get cache size in bytes
    public func cacheSize() throws -> Int64 {
        guard FileManager.default.fileExists(atPath: cacheDirectory.path) else {
            return 0
        }

        let enumerator = FileManager.default.enumerator(at: cacheDirectory, includingPropertiesForKeys: [.fileSizeKey])

        var totalSize: Int64 = 0
        while let fileURL = enumerator?.nextObject() as? URL {
            let attributes = try fileURL.resourceValues(forKeys: [.fileSizeKey])
            totalSize += Int64(attributes.fileSize ?? 0)
        }

        return totalSize
    }
}

// MARK: - Component Paths

/// Paths to all downloaded LTX-2.3 components
public struct LTXComponentPaths: Sendable {
    /// Directory containing Gemma text encoder weights (VLM Gemma, shared)
    public let textEncoderDir: URL
    /// Directory containing tokenizer files (same as textEncoderDir for VLM Gemma)
    public let tokenizerDir: URL
    /// Path to unified weights file (contains transformer + VAE + connector)
    public let unifiedWeightsPath: URL
}

// MARK: - Weight Loader

/// Loads model weights from SafeTensors files
/// Following the Diffusers per-component loading pattern
class LTXWeightLoader {

    // MARK: - Config Parsing

    /// Parse VAE config.json and return whether timestep conditioning is enabled
    /// - Parameter weightsPath: Path to the VAE weights file (config.json is expected in the same directory)
    static func parseVAEConfig(from weightsPath: URL) -> Bool {
        let configPath = weightsPath.deletingLastPathComponent().appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath.path),
              let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let timestepConditioning = json["timestep_conditioning"] as? Bool else {
            LTXDebug.log("VAE config.json not found or missing timestep_conditioning, defaulting to false")
            return false
        }
        LTXDebug.log("VAE config: timestep_conditioning=\(timestepConditioning)")
        return timestepConditioning
    }

    // MARK: - Per-Component Loading

    /// Load transformer weights from the unified safetensors file
    ///
    /// Extracts only transformer keys (those with `model.diffusion_model.` prefix),
    /// strips the prefix, and maps to Swift model format.
    ///
    /// - Parameter path: Path to the unified safetensors file
    /// - Returns: Mapped transformer weights ready to apply
    static func loadTransformerWeights(from path: String, includeAudio: Bool = false) throws -> [String: MLXArray] {
        LTXDebug.log("Loading transformer weights from: \(path)")
        let startTime = Date()

        var allWeights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(allWeights.count) tensors via mmap")

        let diffusionPrefix = "model.diffusion_model."
        let videoConnectorPrefix = "model.diffusion_model.video_embeddings_connector."
        let audioConnectorPrefix = "model.diffusion_model.audio_embeddings_connector."

        var raw: [String: MLXArray] = [:]
        let allKeys = Array(allWeights.keys)
        for key in allKeys {
            // Skip non-transformer keys
            if key.hasSuffix(".weight_scale") || key.hasSuffix(".input_scale") { continue }
            if !includeAudio {
                if key.contains("audio") || key.hasPrefix("vocoder") || key.contains("av_ca_") { continue }
            }
            if !key.hasPrefix(diffusionPrefix) { continue }
            // Connector keys go to text encoder, not transformer
            if key.hasPrefix(videoConnectorPrefix) { continue }
            if key.hasPrefix(audioConnectorPrefix) { continue }

            if let value = allWeights.removeValue(forKey: key) {
                raw[String(key.dropFirst(diffusionPrefix.count))] = value
            }
        }
        // Free remaining keys not used
        allWeights.removeAll()

        let mapped = mapTransformerWeights(raw, includeAudio: includeAudio)
        LTXDebug.log("Extracted \(mapped.count) transformer weights in \(String(format: "%.1f", Date().timeIntervalSince(startTime)))s")
        return mapped
    }

    /// Load VAE decoder weights from standalone safetensors file
    ///
    /// The standalone VAE file has keys without the `vae.` prefix.
    /// Keys with `encoder.` prefix are skipped (we only need the decoder).
    ///
    /// - Parameter path: Path to the VAE safetensors file
    /// - Returns: Mapped VAE weights ready to apply
    static func loadVAEWeights(from path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading VAE weights from: \(path)")

        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(raw.count) tensors")

        // The standalone VAE file keys already lack the "vae." prefix
        // but still have "decoder." prefix which mapVAEWeights strips
        let mapped = mapVAEWeights(raw)
        return mapped
    }

    /// Load connector weights from standalone safetensors file
    ///
    /// The standalone connector file uses Format 1 keys:
    /// - `text_proj_in.*` → feature_extractor
    /// - `video_connector.*` → embeddings_connector
    ///
    /// - Parameter path: Path to the connector safetensors file
    /// - Returns: Mapped text encoder weights ready to apply
    static func loadConnectorWeights(from path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading connector weights from: \(path)")

        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(raw.count) tensors")

        let mapped = mapTextEncoderWeights(raw)
        return mapped
    }

    // MARK: - File Loading

    /// Load all weights from a model directory (multiple safetensors files)
    static func loadWeights(from modelPath: String) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw LTXError.fileNotFound("No safetensors files found in: \(modelPath)")
        }

        LTXDebug.log("Found \(safetensorFiles.count) safetensor files in \(modelPath)")

        var allWeights: [String: MLXArray] = [:]

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))

            for (key, value) in weights {
                allWeights[key] = value
            }

            LTXDebug.log("Loaded \(weights.count) tensors from \(filename)")
        }

        return allWeights
    }

    /// Load weights from URL
    static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        try loadWeights(from: url.path)
    }

    /// Load a single safetensors file
    static func loadSingleFile(path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading safetensors from: \(path)")
        let weights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(weights.count) tensors")
        return weights
    }

    /// Load a single safetensors file from URL
    static func loadSingleFile(url: URL) throws -> [String: MLXArray] {
        try loadSingleFile(path: url.path)
    }

    // MARK: - Weight Mapping

    /// Map Python transformer weight keys to Swift module paths
    ///
    /// Uses `removeValue(forKey:)` to free source weights progressively,
    /// reducing peak memory during loading by ~30%.
    static func mapTransformerWeights(_ weights: [String: MLXArray], includeAudio: Bool = false) -> [String: MLXArray] {
        var source = weights
        var mapped: [String: MLXArray] = [:]

        let allKeys = Array(source.keys)
        for (i, key) in allKeys.enumerated() {
            guard let value = source.removeValue(forKey: key) else { continue }
            if let newKey = mapTransformerKey(key, includeAudio: includeAudio) {
                mapped[newKey] = value
            }
            // Periodic eval to materialize and free intermediate references
            if (i + 1) % 100 == 0 {
                let recent: [MLXArray] = Array(mapped.values.suffix(100))
                eval(recent)
            }
        }

        LTXDebug.log("Mapped \(mapped.count) transformer weights (from \(weights.count) total)")
        return mapped
    }

    /// Map a single transformer key from safetensors to Swift model format
    ///
    /// Returns nil for keys that should be skipped (e.g., audio-related keys when includeAudio=false)
    private static func mapTransformerKey(_ key: String, includeAudio: Bool = false) -> String? {
        // Skip audio-related keys when not in audio mode
        if !includeAudio {
            if key.hasPrefix("audio_") ||
               key.contains(".audio_") ||
               key.hasPrefix("av_cross_attn_") ||
               key.contains("video_to_audio") ||
               key.contains("video_a2v") ||
               key.contains("a2v_ca") ||
               key.contains("scale_shift_table_a2v") {
                return nil
            }
        }

        var k = key

        // Top-level structural mappings (prefix-aware to avoid matching audio_proj_in)
        if k.hasPrefix("proj_in.") {
            k = "patchify_proj." + String(k.dropFirst("proj_in.".count))
        }

        // AdaLN: video time_embed → adaln_single (prefix-aware to avoid matching audio_time_embed)
        if k.hasPrefix("time_embed.emb.timestep_embedder.") {
            k = "adaln_single.emb." + String(k.dropFirst("time_embed.emb.timestep_embedder.".count))
        } else if k.hasPrefix("time_embed.linear.") {
            k = "adaln_single." + String(k.dropFirst("time_embed.".count))
        } else if k.hasPrefix("adaln_single.emb.timestep_embedder.") {
            k = "adaln_single.emb." + String(k.dropFirst("adaln_single.emb.timestep_embedder.".count))
        }

        // General: flatten .emb.timestep_embedder. → .emb. for ALL AdaLayerNormSingle
        // (handles audio_time_embed, av_cross_attn_*, etc.)
        k = k.replacingOccurrences(of: ".emb.timestep_embedder.", with: ".emb.")

        // Attention norms (applies to video and audio)
        k = k.replacingOccurrences(of: ".norm_q.", with: ".q_norm.")
        k = k.replacingOccurrences(of: ".norm_k.", with: ".k_norm.")

        // Remove indexed to_out (applies to video and audio)
        k = k.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

        // FFN mappings (applies to both .ff. and audio_ff.)
        // Use pattern without leading dot so "audio_ff.net." also matches
        k = k.replacingOccurrences(of: "ff.net.0.proj.", with: "ff.project_in.proj.")
        k = k.replacingOccurrences(of: "ff.net.2.", with: "ff.project_out.")

        return k
    }

    /// Map VAE weight keys from safetensors to Swift module paths
    ///
    /// LTX-2.3 format: flat `up_blocks.{0-8}` with dots → `up_blocks_{0-8}` with underscores
    /// Uses `removeValue(forKey:)` to free source weights progressively.
    static func mapVAEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var source = weights
        var mapped: [String: MLXArray] = [:]

        let allKeys = Array(source.keys)
        for key in allKeys {
            guard let value = source.removeValue(forKey: key) else { continue }
            // Skip encoder weights
            if key.hasPrefix("encoder.") { continue }

            // Handle per-channel statistics (unified format)
            if key.contains("per_channel_statistics") {
                let basename = key.components(separatedBy: ".").last ?? ""
                if basename == "mean-of-means" {
                    mapped["mean_of_means"] = value
                } else if basename == "std-of-means" {
                    mapped["std_of_means"] = value
                }
                continue
            }

            var newKey = key

            // Remove decoder. prefix
            if newKey.hasPrefix("decoder.") {
                newKey = String(newKey.dropFirst("decoder.".count))
            }

            // LTX-2.3: flat up_blocks.{i}. → up_blocks_{i}. (9 blocks, 0-8)
            for i in 0...8 {
                let src = "up_blocks.\(i)."
                if newKey.hasPrefix(src) {
                    newKey = "up_blocks_\(i)." + String(newKey.dropFirst(src.count))
                    break
                }
            }

            // "resnets" → "res_blocks" (weight key vs Swift @ModuleInfo key)
            newKey = newKey.replacingOccurrences(of: ".resnets.", with: ".res_blocks.")

            mapped[newKey] = value
        }

        LTXDebug.log("Mapped \(mapped.count) VAE decoder weights (encoder weights skipped)")
        if LTXDebug.isEnabled {
            let sortedKeys = mapped.keys.sorted()
            LTXDebug.log("VAE mapped keys: \(sortedKeys.prefix(10))...")
        }
        return mapped
    }

    /// Map text encoder weight keys from safetensors to Swift module paths
    ///
    /// Handles two key formats:
    ///
    /// **Format 1 — Standalone connector file** (`connectors/diffusion_pytorch_model.safetensors`):
    /// - `text_proj_in.weight` → `feature_extractor.aggregate_embed.weight`
    /// - `video_connector.*` → `embeddings_connector.*`
    ///
    /// **Format 2 — Unified file** (split by prefix):
    /// - `text_embedding_projection.*` → `feature_extractor.*`
    /// - `video_embeddings_connector.*` → `embeddings_connector.*`
    static func mapTextEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var source = weights
        var mapped: [String: MLXArray] = [:]

        let allKeys = Array(source.keys)
        for key in allKeys {
            guard let value = source.removeValue(forKey: key) else { continue }

            var newKey: String? = nil

            // Format 1: Standalone connector file
            if key.hasPrefix("text_proj_in.") {
                newKey = key.replacingOccurrences(of: "text_proj_in.", with: "feature_extractor.aggregate_embed.")
            } else if key.hasPrefix("video_connector.") {
                var k = key.replacingOccurrences(of: "video_connector.", with: "embeddings_connector.")
                k = applyConnectorInternalMapping(k)
                newKey = k
            } else if key.hasPrefix("audio_connector.") {
                var k = key.replacingOccurrences(of: "audio_connector.", with: "audio_embeddings_connector.")
                k = applyConnectorInternalMapping(k)
                newKey = k
            }
            // Format 2: Unified file
            else if key.hasPrefix("text_embedding_projection.") {
                newKey = key.replacingOccurrences(of: "text_embedding_projection.", with: "feature_extractor.")
            } else if key.hasPrefix("video_embeddings_connector.") {
                var k = key.replacingOccurrences(of: "video_embeddings_connector.", with: "embeddings_connector.")
                k = applyConnectorInternalMapping(k)
                newKey = k
            } else if key.hasPrefix("audio_embeddings_connector.") {
                // Audio connector keys already match the Swift model key
                var k = key
                k = applyConnectorInternalMapping(k)
                newKey = k
            }

            if let newKey = newKey {
                mapped[newKey] = value
            }
        }

        LTXDebug.log("Mapped \(mapped.count) text encoder weights")
        return mapped
    }

    /// Apply internal key remapping for connector transformer blocks
    private static func applyConnectorInternalMapping(_ key: String) -> String {
        var k = key
        k = k.replacingOccurrences(of: "transformer_blocks.", with: "transformer_1d_blocks.")
        k = k.replacingOccurrences(of: ".norm_q.", with: ".q_norm.")
        k = k.replacingOccurrences(of: ".norm_k.", with: ".k_norm.")
        k = k.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
        k = k.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.project_in.proj.")
        k = k.replacingOccurrences(of: ".ff.net.2.", with: ".ff.project_out.")
        return k
    }

    // MARK: - Weight Application

    /// Apply weights to a transformer model (video-only or dual video/audio)
    static func applyTransformerWeights(
        _ weights: [String: MLXArray],
        to model: Module,
        includeAudio: Bool = false
    ) throws {
        let mapped: [String: MLXArray]
        // If keys already look mapped (contain patchify_proj or adaln_single), skip re-mapping
        if weights.keys.contains(where: { $0.hasPrefix("patchify_proj.") || $0.hasPrefix("adaln_single.") }) {
            mapped = weights
        } else {
            mapped = mapTransformerWeights(weights, includeAudio: includeAudio)
        }

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        var updates: [String: MLXArray] = [:]
        var notFound = 0

        var unmatchedKeys: [String] = []
        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                unmatchedKeys.append(key)
            }
        }
        if !unmatchedKeys.isEmpty {
            let sorted = unmatchedKeys.sorted()
            let sample = sorted.prefix(10).joined(separator: ", ")
            LTXDebug.log("Transformer: \(unmatchedKeys.count) unmatched keys: \(sample)\(unmatchedKeys.count > 10 ? "..." : "")")
        }

        // Convert float32 parameters to bfloat16 (matching Python behavior)
        var f32Converted = 0
        for (key, value) in updates {
            if value.dtype == .float32 {
                updates[key] = value.asType(.bfloat16)
                f32Converted += 1
            }
        }
        if f32Converted > 0 {
            LTXDebug.log("Converted \(f32Converted) float32 parameters to bfloat16")
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        // Check for model parameters NOT loaded (would keep random initialization!)
        let loadedKeys = Set(updates.keys)
        let missingFromModel = flatParameters.keys.filter { !loadedKeys.contains($0) }.sorted()

        // Log weight stats (always for unmatched/missing, debug for success)
        if !unmatchedKeys.isEmpty || !missingFromModel.isEmpty {
            print("[Weights] Transformer: \(updates.count) loaded, \(notFound) unmatched, \(missingFromModel.count) missing")
            if !unmatchedKeys.isEmpty {
                print("[Weights]   Unmatched: \(unmatchedKeys.sorted().prefix(10).joined(separator: ", "))")
            }
            if !missingFromModel.isEmpty {
                print("[Weights]   Missing: \(missingFromModel.prefix(10).joined(separator: ", "))")
            }
        }
        LTXDebug.log("Applied \(updates.count) weights to transformer (\(notFound) unmatched, \(missingFromModel.count) missing)")
    }

    /// Apply weights to a VAE decoder model
    static func applyVAEWeights(
        _ weights: [String: MLXArray],
        to model: VideoDecoder
    ) throws {
        let mapped: [String: MLXArray]
        // If keys already look mapped (contain up_blocks_), skip re-mapping
        if weights.keys.contains(where: { $0.hasPrefix("up_blocks_") || $0 == "mean_of_means" }) {
            mapped = weights
        } else {
            mapped = mapVAEWeights(weights)
        }

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        var updates: [String: MLXArray] = [:]
        var notFound = 0
        var unmatchedKeys: [String] = []

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                unmatchedKeys.append(key)
                if notFound <= 10 {
                    LTXDebug.log("VAE: No parameter for mapped key: \(key)")
                }
            }
        }

        // Also check for model parameters that were NOT loaded
        let loadedKeys = Set(updates.keys)
        let missingFromModel = flatParameters.keys.filter { !loadedKeys.contains($0) }.sorted()
        if !missingFromModel.isEmpty && LTXDebug.isEnabled {
            LTXDebug.log("VAE: \(missingFromModel.count) model params NOT loaded:")
            for k in missingFromModel.prefix(10) {
                LTXDebug.log("  missing: \(k)")
            }
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))
        LTXDebug.log("Applied \(updates.count) weights to VAE (\(notFound) unmatched)")
    }

    /// Apply weights to a text encoder model
    static func applyTextEncoderWeights(
        _ weights: [String: MLXArray],
        to model: VideoGemmaTextEncoderModel
    ) throws {
        let mapped: [String: MLXArray]
        // If keys already look mapped (contain feature_extractor. or embeddings_connector.), skip
        if weights.keys.contains(where: { $0.hasPrefix("feature_extractor.") || $0.hasPrefix("embeddings_connector.") }) {
            mapped = weights
        } else {
            mapped = mapTextEncoderWeights(weights)
        }

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        let hasAudioConnector = model.audioEmbeddingsConnector != nil

        var updates: [String: MLXArray] = [:]
        var notFound = 0
        var skippedAudio = 0

        for (key, value) in mapped {
            // Skip audio connector keys when audio connector is not present
            if !hasAudioConnector && key.hasPrefix("audio_embeddings_connector.") {
                skippedAudio += 1
                continue
            }
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 5 {
                    LTXDebug.log("TextEncoder: No parameter for key: \(key)")
                }
            }
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))
        if skippedAudio > 0 {
            LTXDebug.log("Applied \(updates.count) weights to TextEncoder (skipped \(skippedAudio) audio connector keys)")
        } else {
            LTXDebug.log("Applied \(updates.count) weights to TextEncoder (\(notFound) unmatched)")
        }
    }

    // MARK: - Audio VAE + Vocoder Weight Loading

    /// Load Audio VAE weights from safetensors file
    ///
    /// Extracts decoder and latent stat keys, applying Conv2d weight transposition.
    /// When includeEncoder is true, also loads encoder weights for A2Vid.
    static func loadAudioVAEWeights(from path: String, includeEncoder: Bool = false) throws -> [String: MLXArray] {
        LTXDebug.log("Loading Audio VAE weights from: \(path)")
        let raw = try loadArrays(url: URL(fileURLWithPath: path))

        var filteredWeights: [String: MLXArray] = [:]
        for (key, value) in raw {
            if key.hasPrefix("decoder.") || key == "latents_mean" || key == "latents_std" {
                filteredWeights[key] = value
            } else if includeEncoder && key.hasPrefix("encoder.") {
                filteredWeights[key] = value
            }
        }

        let encoderCount = filteredWeights.keys.filter { $0.hasPrefix("encoder.") }.count
        LTXDebug.log("Audio VAE: \(filteredWeights.count) weights (\(encoderCount) encoder)")
        return filteredWeights
    }

    /// Apply weights to an AudioVAE model
    static func applyAudioVAEWeights(
        _ weights: [String: MLXArray],
        to model: AudioVAE
    ) throws {
        // Sanitize: transpose Conv2d weights from PyTorch to MLX format
        let sanitized = model.sanitize(weights: weights)

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updates: [String: MLXArray] = [:]
        var notFound = 0

        for (key, value) in sanitized {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 5 {
                    LTXDebug.log("AudioVAE: No parameter for key: \(key)")
                }
            }
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model.parameters())
        LTXDebug.log("Applied \(updates.count) weights to AudioVAE (\(notFound) unmatched)")
    }

    /// Load Vocoder weights from safetensors file
    static func loadVocoderWeights(from path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading Vocoder weights from: \(path)")
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Vocoder: \(raw.count) weights loaded")
        return raw
    }

    /// Apply weights to an LTX2Vocoder model
    static func applyVocoderWeights(
        _ weights: [String: MLXArray],
        to model: LTX2Vocoder
    ) throws {
        // Sanitize: transpose Conv1d and ConvTranspose1d weights
        let sanitized = model.sanitize(weights: weights)

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updates: [String: MLXArray] = [:]
        var notFound = 0

        for (key, value) in sanitized {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 5 {
                    LTXDebug.log("Vocoder: No parameter for key: \(key)")
                }
            }
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model.parameters())
        LTXDebug.log("Applied \(updates.count) weights to Vocoder (\(notFound) unmatched)")
    }

    // MARK: - VAE Encoder Weight Loading

    /// Load VAE encoder weights from the standalone VAE safetensors file
    ///
    /// Extracts keys with `encoder.` prefix (which are skipped by loadVAEWeights).
    ///
    /// - Parameter path: Path to the VAE safetensors file
    /// - Returns: Mapped encoder weights ready to apply
    static func loadVAEEncoderWeights(from path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading VAE encoder weights from: \(path)")

        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(raw.count) total VAE tensors")

        let mapped = mapVAEEncoderWeights(raw)
        return mapped
    }

    /// Map VAE encoder weight keys from Diffusers safetensors to Swift module paths
    ///
    /// Diffusers encoder structure:
    ///   encoder.conv_in.* -> conv_in.*
    ///   encoder.down_blocks.{i}.resnets.{j}.* -> down_blocks_{i}.resnets.resnets.{j}.*  (WRONG)
    ///   encoder.down_blocks.{i}.downsamplers.0.* -> down_blocks_{i}.downsamplers.*
    ///   encoder.mid_block.resnets.{j}.* -> mid_block.resnets.{j}.*
    ///   encoder.conv_out.* -> conv_out.*
    static func mapVAEEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var source = weights
        var mapped: [String: MLXArray] = [:]

        // Safetensors uses flat block indexing (0-8) for the encoder:
        //   0=res(128,4), 1=downsample, 2=res(256,6), 3=downsample,
        //   4=res(512,4), 5=downsample, 6=res(1024,2), 7=downsample, 8=mid_res(1024,2)
        //
        // Swift model uses grouped blocks:
        //   down_blocks_0 (resnets + downsamplers), down_blocks_1, ..., down_blocks_3, mid_block
        //
        // Mapping: flat even indices → resnets, flat odd indices → downsamplers
        //   flat 0 → down_blocks_0.resnets, flat 1 → down_blocks_0.downsamplers
        //   flat 2 → down_blocks_1.resnets, flat 3 → down_blocks_1.downsamplers
        //   flat 4 → down_blocks_2.resnets, flat 5 → down_blocks_2.downsamplers
        //   flat 6 → down_blocks_3.resnets, flat 7 → down_blocks_3.downsamplers
        //   flat 8 → mid_block

        let allKeys = Array(source.keys)
        for key in allKeys {
            guard let value = source.removeValue(forKey: key) else { continue }

            // Only process encoder.* keys
            guard key.hasPrefix("encoder.") else { continue }

            var newKey = String(key.dropFirst("encoder.".count))
            let newValue = value
            // No weight transposition: Conv3dFull stores weights in PyTorch format (O,I,T,H,W)
            // and handles the conversion internally during forward pass.

            // Handle flat down_blocks indexing
            var handled = false
            for flatIdx in 0...8 {
                let prefix = "down_blocks.\(flatIdx)."
                guard newKey.hasPrefix(prefix) else { continue }
                let suffix = String(newKey.dropFirst(prefix.count))

                if flatIdx == 8 {
                    // Block 8 = mid_block
                    // res_blocks.{j}.* → mid_block.resnets.{j}.*
                    if suffix.hasPrefix("res_blocks.") {
                        let resSuffix = String(suffix.dropFirst("res_blocks.".count))
                        newKey = "mid_block.resnets.\(resSuffix)"
                    } else {
                        newKey = "mid_block.\(suffix)"
                    }
                } else if flatIdx % 2 == 0 {
                    // Even flat indices = resblock groups
                    let groupIdx = flatIdx / 2
                    // res_blocks.{j}.* → down_blocks_{group}.resnets.resnets.{j}.*
                    if suffix.hasPrefix("res_blocks.") {
                        let resSuffix = String(suffix.dropFirst("res_blocks.".count))
                        newKey = "down_blocks_\(groupIdx).resnets.resnets.\(resSuffix)"
                    } else {
                        newKey = "down_blocks_\(groupIdx).resnets.\(suffix)"
                    }
                } else {
                    // Odd flat indices = downsamplers
                    let groupIdx = flatIdx / 2
                    // conv.* → down_blocks_{group}.downsamplers.conv.*
                    newKey = "down_blocks_\(groupIdx).downsamplers.\(suffix)"
                }
                handled = true
                break
            }

            if !handled {
                // conv_in.*, conv_out.*, mid_block.* pass through unchanged
            }

            mapped[newKey] = newValue
        }

        LTXDebug.log("Mapped \(mapped.count) VAE encoder weights")
        if LTXDebug.isEnabled {
            let sortedKeys = mapped.keys.sorted()
            LTXDebug.log("VAE encoder mapped keys: \(sortedKeys.prefix(10))...")
        }
        return mapped
    }

    /// Apply weights to a VAE encoder model
    static func applyVAEEncoderWeights(
        _ weights: [String: MLXArray],
        to model: VideoEncoder
    ) throws {
        let mapped: [String: MLXArray]
        // If keys already look mapped (contain down_blocks_), skip re-mapping
        if weights.keys.contains(where: { $0.hasPrefix("down_blocks_") || $0.hasPrefix("conv_in.") }) {
            mapped = weights
        } else {
            mapped = mapVAEEncoderWeights(weights)
        }

        let flatParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        var updates: [String: MLXArray] = [:]
        var notFound = 0
        var unmatchedKeys: [String] = []

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                unmatchedKeys.append(key)
                if notFound <= 10 {
                    LTXDebug.log("VAE Encoder: No parameter for mapped key: \(key)")
                }
            }
        }

        // Check for model parameters that were NOT loaded
        let loadedKeys = Set(updates.keys)
        let missingFromModel = flatParameters.keys.filter { !loadedKeys.contains($0) }.sorted()
        if !missingFromModel.isEmpty && LTXDebug.isEnabled {
            LTXDebug.log("VAE Encoder: \(missingFromModel.count) model params NOT loaded:")
            for k in missingFromModel.prefix(10) {
                LTXDebug.log("  missing: \(k)")
            }
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))
        LTXDebug.log("Applied \(updates.count) weights to VAE Encoder (\(notFound) unmatched)")
    }

    /// Load VAE encoder weights from the unified safetensors file
    ///
    /// Extracts keys with `vae.encoder.` prefix from the unified file.
    static func loadVAEEncoderWeightsFromUnified(from path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading VAE encoder weights from unified file: \(path)")

        let allWeights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(allWeights.count) total tensors")

        // Extract vae.encoder.* keys and strip "vae." prefix so they become "encoder.*"
        // which mapVAEEncoderWeights expects
        var encoderWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("vae.encoder.") {
                encoderWeights["encoder." + String(key.dropFirst("vae.encoder.".count))] = value
            }
        }

        let mapped = mapVAEEncoderWeights(encoderWeights)
        LTXDebug.log("Extracted \(mapped.count) VAE encoder weights from unified file")
        return mapped
    }

    // MARK: - Unified File Splitting

    /// Split a unified weights file into transformer, VAE, and connector components
    ///
    /// Loads the file once and classifies each key by prefix:
    /// - `model.diffusion_model.*` (excluding connector) → transformer
    /// - `vae.*` + `per_channel_statistics.*` → VAE decoder
    /// - `model.diffusion_model.video_embeddings_connector.*` + `text_embedding_projection.*` → connector
    ///
    /// - Parameter path: Path to the unified safetensors file
    /// - Returns: Tuple of (transformer, vae, connector) mapped weights
    static func splitUnifiedWeightsFile(path: String, includeAudio: Bool = false) throws -> (transformer: [String: MLXArray], vae: [String: MLXArray], connector: [String: MLXArray]) {
        LTXDebug.log("Splitting unified weights from: \(path)")
        let allWeights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(allWeights.count) tensors from unified file")
        return splitUnifiedWeightsDict(allWeights, includeAudio: includeAudio)
    }

    /// Split a pre-loaded unified weights dictionary into components
    ///
    /// Uses `removeValue(forKey:)` to free source weights progressively.
    static func splitUnifiedWeightsDict(_ allWeights: [String: MLXArray], includeAudio: Bool = false) -> (transformer: [String: MLXArray], vae: [String: MLXArray], connector: [String: MLXArray]) {
        let diffusionPrefix = "model.diffusion_model."
        let videoConnectorPrefix = "model.diffusion_model.video_embeddings_connector."
        let audioConnectorPrefix = "model.diffusion_model.audio_embeddings_connector."
        let projPrefix = "model.diffusion_model.text_embedding_projection."

        var source = allWeights
        var transformerRaw: [String: MLXArray] = [:]
        var vaeRaw: [String: MLXArray] = [:]
        var connectorRaw: [String: MLXArray] = [:]

        let allKeys = Array(source.keys)
        for key in allKeys {
            guard let value = source.removeValue(forKey: key) else { continue }
            // Skip FP8 scale keys
            if key.hasSuffix(".weight_scale") || key.hasSuffix(".input_scale") { continue }
            // Skip audio keys when not in audio mode
            if !includeAudio {
                if key.contains("audio") || key.hasPrefix("vocoder") || key.contains("av_ca_") { continue }
            }

            if key.hasPrefix(videoConnectorPrefix) {
                connectorRaw["video_embeddings_connector." + String(key.dropFirst(videoConnectorPrefix.count))] = value
            } else if includeAudio && key.hasPrefix(audioConnectorPrefix) {
                // Audio connector keys → connector bucket
                connectorRaw["audio_embeddings_connector." + String(key.dropFirst(audioConnectorPrefix.count))] = value
            } else if key.hasPrefix(projPrefix) {
                connectorRaw["text_embedding_projection." + String(key.dropFirst(projPrefix.count))] = value
            } else if key.hasPrefix("text_embedding_projection.") {
                // Top-level text_embedding_projection (without model.diffusion_model. prefix)
                connectorRaw[key] = value
            } else if key.hasPrefix(diffusionPrefix) {
                transformerRaw[String(key.dropFirst(diffusionPrefix.count))] = value
            } else if key.hasPrefix("vae.") {
                vaeRaw[String(key.dropFirst("vae.".count))] = value
            } else if key.contains("per_channel_statistics") {
                vaeRaw[key] = value
            }
        }

        let transformer = mapTransformerWeights(transformerRaw, includeAudio: includeAudio)
        let vae = mapVAEWeights(vaeRaw)
        let connector = mapTextEncoderWeights(connectorRaw)

        LTXDebug.log("Split unified: \(allWeights.count) input → transformer=\(transformer.count), vae=\(vae.count), connector=\(connector.count)")
        return (transformer, vae, connector)
    }

    /// Get summary of loaded weights
    static func summarizeWeights(_ weights: [String: MLXArray]) {
        var totalParams: Int64 = 0
        var byPrefix: [String: Int64] = [:]

        for (key, array) in weights {
            let params = Int64(array.shape.reduce(1, *))
            totalParams += params

            let prefix = String(key.split(separator: ".").first ?? Substring(key))
            byPrefix[prefix, default: 0] += params
        }

        LTXDebug.log("Weight Summary:")
        for (prefix, params) in byPrefix.sorted(by: { $0.value > $1.value }) {
            let gb = Float(params * 2) / 1_000_000_000
            LTXDebug.log("  \(prefix): \(params) params (~\(String(format: "%.2f", gb))GB)")
        }
        LTXDebug.log("Total: \(totalParams) parameters")
    }
}

// Need to import MLX for loadArrays function
@preconcurrency import MLX
import MLXNN
