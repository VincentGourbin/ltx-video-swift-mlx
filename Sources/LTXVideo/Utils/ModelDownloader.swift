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
public actor ModelDownloader {
    /// HuggingFace token for accessing gated models
    private let hfToken: String?

    /// Base cache directory
    private let cacheDirectory: URL

    /// URLSession for downloads
    private let session: URLSession

    public init(hfToken: String? = nil, cacheDir: URL? = nil) {
        self.hfToken = hfToken

        // Use custom cache directory or default
        if let customDir = cacheDir {
            self.cacheDirectory = customDir
        } else {
            let cacheBase = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            self.cacheDirectory = cacheBase.appendingPathComponent("models")
        }

        // Create session with configuration
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600  // 1 hour timeout for large files
        self.session = URLSession(configuration: config)
    }

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
        // Skip if file already exists with correct size
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

    /// Repository for Gemma text encoder (MLX optimized)
    public static let gemmaRepo = "mlx-community/gemma-3-12b-it-4bit"

    /// Download Gemma text encoder for LTX-2
    ///
    /// - Parameter progress: Optional progress callback
    /// - Returns: Path to the downloaded Gemma model directory
    public func downloadGemma(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = Self.gemmaRepo

        progress?(DownloadProgress(
            progress: 0.0,
            message: "Preparing to download Gemma 3 12B (MLX 4-bit)..."
        ))

        let gemmaDir = cacheDirectory.appendingPathComponent("gemma-3-12b-mlx")

        // Get file list from HuggingFace
        let files = try await listRepoFiles(repoId: repoId)

        // Filter to required files
        let requiredFiles = files.filter { file in
            file.hasSuffix(".safetensors") ||
            file == "config.json" ||
            file == "tokenizer.json" ||
            file == "tokenizer_config.json" ||
            file == "special_tokens_map.json" ||
            file == "tokenizer.model"
        }

        // Check if already downloaded
        let allExist = requiredFiles.allSatisfy { file in
            FileManager.default.fileExists(atPath: gemmaDir.appendingPathComponent(file).path)
        }

        if allExist {
            progress?(DownloadProgress(
                progress: 1.0,
                message: "Gemma already downloaded"
            ))
            return gemmaDir
        }

        // Create directory
        try FileManager.default.createDirectory(at: gemmaDir, withIntermediateDirectories: true)

        // Download each file
        for (index, file) in requiredFiles.enumerated() {
            let fileProgress = Double(index) / Double(requiredFiles.count)
            progress?(DownloadProgress(
                progress: fileProgress,
                currentFile: file,
                message: "Downloading \(file)..."
            ))

            try await downloadFile(
                repoId: repoId,
                filename: file,
                to: gemmaDir.appendingPathComponent(file)
            )
        }

        progress?(DownloadProgress(
            progress: 1.0,
            message: "Gemma download complete"
        ))

        return gemmaDir
    }

    /// Check if Gemma is downloaded
    public func isGemmaDownloaded() -> Bool {
        let gemmaDir = cacheDirectory.appendingPathComponent("gemma-3-12b-mlx")
        // Check for essential file
        return FileManager.default.fileExists(atPath: gemmaDir.appendingPathComponent("config.json").path)
    }

    /// Get the Gemma model directory (downloads if needed)
    public func getGemmaPath(
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        return try await downloadGemma(progress: progress)
    }

    // MARK: - LTX Unified Weights

    /// Download unified LTX-2 weights (single safetensors file containing all components)
    ///
    /// The file contains transformer, VAE, text encoder, and connector weights.
    /// This matches the Python LTX-2-MLX reference implementation's loading approach.
    ///
    /// - Parameters:
    ///   - model: The model variant to download
    ///   - progress: Optional progress callback
    /// - Returns: Path to the downloaded safetensors file
    public func downloadLTXWeights(
        model: LTXModel,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        let repoId = model.huggingFaceRepo
        let filename = model.weightsFilename

        progress?(DownloadProgress(
            progress: 0.0,
            message: "Preparing to download \(model.displayName) weights..."
        ))

        let weightsDir = cacheDirectory.appendingPathComponent("ltx-weights")
        let destination = weightsDir.appendingPathComponent(filename)

        // Check if already downloaded
        if FileManager.default.fileExists(atPath: destination.path) {
            progress?(DownloadProgress(
                progress: 1.0,
                message: "LTX weights already downloaded"
            ))
            return destination
        }

        // Create directory
        try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)

        // Download the unified file
        progress?(DownloadProgress(
            progress: 0.1,
            currentFile: filename,
            message: "Downloading \(filename)..."
        ))

        try await downloadFile(repoId: repoId, filename: filename, to: destination)

        progress?(DownloadProgress(
            progress: 1.0,
            message: "LTX weights download complete"
        ))

        return destination
    }

    /// Check if unified LTX weights are downloaded
    public func isLTXWeightsDownloaded(_ model: LTXModel) -> Bool {
        let weightsDir = cacheDirectory.appendingPathComponent("ltx-weights")
        let destination = weightsDir.appendingPathComponent(model.weightsFilename)
        return FileManager.default.fileExists(atPath: destination.path)
    }

    /// Get the LTX weights path (downloads if needed)
    public func getLTXWeightsPath(
        model: LTXModel,
        progress: DownloadProgressCallback? = nil
    ) async throws -> URL {
        return try await downloadLTXWeights(model: model, progress: progress)
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

// MARK: - Weight Loader

/// Weights split by component from a unified LTX-2 safetensors file
public struct LTXComponentWeights {
    /// Transformer weights (keys with `model.diffusion_model.` prefix stripped)
    public var transformer: [String: MLXArray]
    /// VAE decoder weights (keys with `vae.` prefix stripped, encoder excluded)
    public var vaeDecoder: [String: MLXArray]
    /// Text encoder weights (feature extractor + connector)
    public var textEncoder: [String: MLXArray]
}

/// Loads model weights from SafeTensors files
/// Following the exact pattern from flux-mlx-swift
public class LTXWeightLoader {

    // MARK: - Unified Weight Loading

    // MARK: - FP8 E4M3FN Dequantization

    /// Lookup table for FP8 E4M3FN (1 sign, 4 exponent, 3 mantissa) → Float32
    ///
    /// The safetensors file stores FP8 weights as uint8. Each byte is a bit pattern
    /// in the E4M3FN format that must be properly interpreted (NOT simply cast to float).
    private static let fp8E4M3FNLookupTable: [Float] = {
        var lut = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            let sign = (i >> 7) & 1
            let exponent = (i >> 3) & 0xF
            let mantissa = i & 0x7
            let signMul: Float = sign == 0 ? 1.0 : -1.0

            if exponent == 0 {
                // Subnormal: (-1)^sign * 2^(-6) * (mantissa / 8)
                lut[i] = signMul * (1.0 / 64.0) * (Float(mantissa) / 8.0)
            } else if exponent == 15 && mantissa == 7 {
                // NaN (0x7F and 0xFF)
                lut[i] = 0.0  // Treat NaN as zero for weight loading
            } else {
                // Normal: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
                lut[i] = signMul * powf(2.0, Float(exponent) - 7.0) * (1.0 + Float(mantissa) / 8.0)
            }
        }
        return lut
    }()

    /// Dequantize a uint8 FP8 E4M3FN weight tensor to float16
    ///
    /// Precomputed FP8 E4M3FN lookup table as MLXArray (256 float32 values)
    /// Maps each uint8 bit pattern (0-255) to its decoded float value.
    /// Initialized once, kept on GPU for fast gather operations.
    private static let fp8LUTArray: MLXArray = {
        var lut = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            let signBit = (i >> 7) & 1
            let exp = (i >> 3) & 0xF
            let mantissa = i & 0x7
            let sign: Float = signBit == 0 ? 1.0 : -1.0

            if exp == 15 && mantissa == 7 {
                // NaN → 0
                lut[i] = 0.0
            } else if exp == 0 {
                // Subnormal: (-1)^s * 2^(-6) * (m/8)
                lut[i] = sign * Float(mantissa) / 512.0
            } else {
                // Normal: (-1)^s * 2^(e-7) * (1 + m/8)
                lut[i] = sign * powf(2.0, Float(exp) - 7.0) * (1.0 + Float(mantissa) / 8.0)
            }
        }
        let arr = MLXArray(lut)
        eval(arr)
        return arr
    }()

    /// Dequantize FP8 E4M3FN weights using CPU-stream gather (LUT lookup)
    ///
    /// FP8 E4M3FN only has 256 possible values. We precompute them in a lookup
    /// table and use MLX's `take` (gather) operation on the CPU stream.
    /// CPU stream is required because memory-mapped safetensors data can cause
    /// GPU page faults (Metal timeout) on Apple Silicon — the CPU MMU handles
    /// page faults transparently while the GPU cannot.
    ///
    /// - Parameters:
    ///   - weight: uint8 tensor containing FP8 E4M3FN bit patterns
    ///   - scale: Per-tensor scale factor
    /// - Returns: float16 tensor with properly decoded values (already evaluated)
    /// Dequantize a batch of FP8 weights efficiently.
    /// 1. Batch-eval raw mmap'd tensors (CPU Load, no GPU)
    /// 2. Queue GPU dequant ops for all tensors
    /// 3. Batch-eval GPU results
    static func dequantizeFP8Batch(_ weights: [(MLXArray, Float)], batchIndex: Int = 0) -> [MLXArray] {
        let t0 = Date()

        // Step 1: Materialize all raw weights from mmap (CPU Load)
        eval(weights.map { $0.0 })
        let t1 = Date()

        // Step 2: Queue GPU dequant for all weights (lazy)
        let results = weights.map { (weight, scale) -> MLXArray in
            let shape = weight.shape
            let indices = weight.flattened().asType(.int32)
            let decoded = fp8LUTArray.take(indices, axis: 0)
            let scaled = decoded * scale
            return scaled.asType(.float16).reshaped(shape)
        }
        let t2 = Date()

        // Step 3: Execute all GPU work at once
        eval(results)
        let t3 = Date()

        if batchIndex < 5 {
            let shapes = weights.map { "\($0.0.shape)" }.joined(separator: ", ")
            LTXDebug.log("    batch \(batchIndex): load=\(String(format: "%.2f", t1.timeIntervalSince(t0)))s, queue=\(String(format: "%.4f", t2.timeIntervalSince(t1)))s, gpu=\(String(format: "%.2f", t3.timeIntervalSince(t2)))s — shapes: \(shapes)")
        }

        return results
    }

    /// Load and split a unified LTX-2 safetensors file into component weights
    ///
    /// The unified file (e.g., `ltx-2-19b-distilled-fp8.safetensors`) contains all weights:
    /// - `model.diffusion_model.*` → transformer
    /// - `vae.*` → VAE encoder/decoder
    /// - `text_embedding_projection.*` → text encoder feature extractor
    /// - `model.diffusion_model.video_embeddings_connector.*` → text encoder connector
    /// - `model.diffusion_model.caption_projection.*` → transformer (caption projection)
    ///
    /// For FP8 models, weights are dequantized using their associated `*_scale` values.
    ///
    /// - Parameters:
    ///   - path: Path to the unified safetensors file
    ///   - isFP8: Whether to apply FP8 dequantization
    /// - Returns: Component weights ready to be mapped and applied
    /// Precomputed FP8 LUT as Swift Float16 array for direct CPU-side dequantization.
    /// This avoids MLX's lazy loading overhead entirely.
    private static let fp8LUTFloat16: [Float16] = {
        var lut = [Float16](repeating: 0, count: 256)
        for i in 0..<256 {
            let signBit = (i >> 7) & 1
            let exp = (i >> 3) & 0xF
            let mantissa = i & 0x7
            let sign: Float = signBit == 0 ? 1.0 : -1.0
            if exp == 15 && mantissa == 7 {
                lut[i] = Float16(0.0)
            } else if exp == 0 {
                lut[i] = Float16(sign * Float(mantissa) / 512.0)
            } else {
                lut[i] = Float16(sign * powf(2.0, Float(exp) - 7.0) * (1.0 + Float(mantissa) / 8.0))
            }
        }
        return lut
    }()

    /// Load unified safetensors file and split into component weights.
    ///
    /// For non-FP8 models (bf16/f16): uses MLX's native `loadArrays(url:)` with mmap,
    /// following the same approach as Flux Swift MLX. This avoids reading the entire
    /// file into memory (e.g., 40GB for distilled model).
    ///
    /// For FP8 models: uses custom parser with CPU LUT dequantization.
    ///
    /// Applies progressive weight removal (Flux pattern) to minimize peak memory:
    /// weights are removed from the input dict as they're classified.
    public static func loadUnifiedWeights(from path: String, isFP8: Bool = false) throws -> LTXComponentWeights {
        LTXDebug.log("Loading unified weights from: \(path)")
        let startTime = Date()

        if isFP8 {
            return try loadUnifiedWeightsFP8(from: path)
        }

        // === Non-FP8 path: use MLX native mmap loading (like Flux) ===
        let loadStart = Date()
        var allWeights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(allWeights.count) tensors via mmap in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")

        // Progressive classification with removal (Flux pattern: reduce peak memory)
        var transformer: [String: MLXArray] = [:]
        var vae: [String: MLXArray] = [:]
        var textEncoder: [String: MLXArray] = [:]
        let diffusionPrefix = "model.diffusion_model."
        let connectorPrefix = "model.diffusion_model.video_embeddings_connector."
        let vaePrefix = "vae."
        let textEmbedPrefix = "text_embedding_projection."

        let allKeys = Array(allWeights.keys)
        for key in allKeys {
            // Skip metadata/scale keys
            if key.hasSuffix(".weight_scale") || key.hasSuffix(".input_scale") { continue }
            if key.contains("audio") || key.hasPrefix("vocoder") || key.contains("av_ca_") { continue }

            guard var value = allWeights.removeValue(forKey: key) else { continue }

            // Convert bfloat16 → float16 (direct, no f32 intermediate — saves memory)
            if value.dtype == .bfloat16 {
                value = value.asType(.float16)
            }

            // Classify into component
            classifyWeight(key: key, weight: value, diffusionPrefix: diffusionPrefix,
                         connectorPrefix: connectorPrefix, vaePrefix: vaePrefix,
                         textEmbedPrefix: textEmbedPrefix,
                         transformer: &transformer, vae: &vae, textEncoder: &textEncoder)
        }
        // allWeights is now mostly empty — memory freed progressively

        LTXDebug.log("Split unified weights: transformer=\(transformer.count), vae=\(vae.count), textEncoder=\(textEncoder.count)")
        LTXDebug.log("[TIME] Load + split weights: \(String(format: "%.1f", Date().timeIntervalSince(startTime)))s")

        return LTXComponentWeights(
            transformer: transformer,
            vaeDecoder: vae,
            textEncoder: textEncoder
        )
    }

    /// FP8 loading path: custom safetensors parser with CPU LUT dequantization
    private static func loadUnifiedWeightsFP8(from path: String) throws -> LTXComponentWeights {
        let readStart = Date()

        // Read entire file into memory (needed for CPU-side FP8 decode)
        let fileData = try Data(contentsOf: URL(fileURLWithPath: path))
        LTXDebug.log("Read \(fileData.count / 1_000_000)MB in \(String(format: "%.1f", Date().timeIntervalSince(readStart)))s")

        // Parse safetensors header
        guard fileData.count >= 8 else {
            throw LTXError.weightLoadingFailed("Safetensors file too small")
        }
        let headerLength: UInt64 = fileData.withUnsafeBytes { buf in
            buf.load(fromByteOffset: 0, as: UInt64.self)
        }
        let headerEnd = 8 + Int(headerLength)
        guard headerEnd <= fileData.count else {
            throw LTXError.weightLoadingFailed("Safetensors header length exceeds file size")
        }
        let headerData = fileData[8..<headerEnd]
        guard let headerJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw LTXError.weightLoadingFailed("Failed to parse safetensors header JSON")
        }
        let dataBaseOffset = headerEnd

        // Collect FP8 scale factors
        var fp8Scales: [String: Float] = [:]
        for (key, info) in headerJSON {
            guard key.hasSuffix(".weight_scale"),
                  let meta = info as? [String: Any],
                  let offsets = meta["data_offsets"] as? [Int],
                  offsets.count >= 2 else { continue }
            let start = dataBaseOffset + offsets[0]
            let end = dataBaseOffset + offsets[1]
            guard end <= fileData.count, end - start == 4 else { continue }
            let scaleValue: Float = fileData.withUnsafeBytes { buf in
                buf.load(fromByteOffset: start, as: Float.self)
            }
            fp8Scales[String(key.dropLast("_scale".count))] = scaleValue
        }
        LTXDebug.log("Found \(fp8Scales.count) FP8 scale factors")

        func mlxDtype(_ dtype: String) -> DType? {
            switch dtype {
            case "F32": return .float32; case "F16": return .float16; case "BF16": return .bfloat16
            case "I64": return .int64; case "I32": return .int32; case "I16": return .int16
            case "I8": return .int8; case "U8", "F8_E4M3": return .uint8; case "BOOL": return DType.bool
            default: return nil
            }
        }

        var transformer: [String: MLXArray] = [:]
        var vae: [String: MLXArray] = [:]
        var textEncoder: [String: MLXArray] = [:]
        let diffusionPrefix = "model.diffusion_model."
        let connectorPrefix = "model.diffusion_model.video_embeddings_connector."
        let vaePrefix = "vae."
        let textEmbedPrefix = "text_embedding_projection."
        var dequantCount = 0

        for (key, info) in headerJSON {
            guard let meta = info as? [String: Any],
                  let dtypeStr = meta["dtype"] as? String,
                  let shape = meta["shape"] as? [Int],
                  let offsets = meta["data_offsets"] as? [Int],
                  offsets.count >= 2 else { continue }

            if key.hasSuffix(".weight_scale") || key.hasSuffix(".input_scale") { continue }
            if key.contains("audio") || key.hasPrefix("vocoder") || key.contains("av_ca_") { continue }

            let start = dataBaseOffset + offsets[0]
            let end = dataBaseOffset + offsets[1]
            guard end <= fileData.count else { continue }

            let weight: MLXArray
            if let scale = fp8Scales[key], dtypeStr == "F8_E4M3" {
                let numElements = shape.reduce(1, *)
                let scaleF16 = Float16(scale)
                let float16Data = fileData.withUnsafeBytes { buf -> [Float16] in
                    let ptr = buf.baseAddress!.advanced(by: start).assumingMemoryBound(to: UInt8.self)
                    var result = [Float16](repeating: 0, count: numElements)
                    for i in 0..<numElements {
                        result[i] = fp8LUTFloat16[Int(ptr[i])] * scaleF16
                    }
                    return result
                }
                weight = MLXArray(float16Data, shape)
                dequantCount += 1
            } else {
                guard let dtype = mlxDtype(dtypeStr) else { continue }
                let tensorData = Data(fileData[start..<end])
                let raw = MLXArray(tensorData, shape, dtype: dtype)
                weight = (dtype == .bfloat16) ? raw.asType(.float16) : raw
            }

            classifyWeight(key: key, weight: weight, diffusionPrefix: diffusionPrefix,
                         connectorPrefix: connectorPrefix, vaePrefix: vaePrefix,
                         textEmbedPrefix: textEmbedPrefix,
                         transformer: &transformer, vae: &vae, textEncoder: &textEncoder)
        }

        LTXDebug.log("Dequantized \(dequantCount) FP8 weights in \(String(format: "%.1f", Date().timeIntervalSince(readStart)))s")
        LTXDebug.log("Split unified weights: transformer=\(transformer.count), vae=\(vae.count), textEncoder=\(textEncoder.count)")

        return LTXComponentWeights(
            transformer: transformer,
            vaeDecoder: vae,
            textEncoder: textEncoder
        )
    }

    /// Split a dictionary of weights from a unified file into components
    ///
    /// Handles FP8 dequantization: weights with a matching `*_scale` key are
    /// converted to float32 and multiplied by their scale factor.
    public static func splitUnifiedWeights(_ allWeights: [String: MLXArray], isFP8: Bool = false) -> LTXComponentWeights {
        var transformer: [String: MLXArray] = [:]
        var vae: [String: MLXArray] = [:]
        var textEncoder: [String: MLXArray] = [:]

        // Collect FP8 scale factors if present
        var fp8Scales: [String: Float] = [:]
        if isFP8 {
            // Batch-eval all scale values at once (they're tiny scalar tensors)
            let scaleEntries = allWeights.filter { $0.key.hasSuffix(".weight_scale") }
            eval(Array(scaleEntries.values))
            for (key, value) in scaleEntries {
                let weightKey = String(key.dropLast("_scale".count))
                fp8Scales[weightKey] = value.item(Float.self)
            }
            if !fp8Scales.isEmpty {
                LTXDebug.log("Found \(fp8Scales.count) FP8 scale factors")
            }
        }

        let diffusionPrefix = "model.diffusion_model."
        let connectorPrefix = "model.diffusion_model.video_embeddings_connector."
        let vaePrefix = "vae."
        let textEmbedPrefix = "text_embedding_projection."

        // First pass: collect FP8 tensors that need dequantization and sort by key
        // Also collect non-FP8 tensors into their destinations
        var fp8Batch: [(key: String, weight: MLXArray, scale: Float)] = []
        let dequantStart = Date()

        for (key, value) in allWeights {
            // Skip metadata keys
            if key.hasSuffix(".weight_scale") || key.hasSuffix(".input_scale") { continue }
            if key.contains("audio") || key.hasPrefix("vocoder") || key.contains("av_ca_") { continue }

            if isFP8, let scale = fp8Scales[key] {
                fp8Batch.append((key: key, weight: value, scale: scale))
            } else {
                // Non-FP8: assign directly (lazy mmap'd reference)
                classifyWeight(key: key, weight: value, diffusionPrefix: diffusionPrefix,
                             connectorPrefix: connectorPrefix, vaePrefix: vaePrefix,
                             textEmbedPrefix: textEmbedPrefix,
                             transformer: &transformer, vae: &vae, textEncoder: &textEncoder)
            }
        }

        // Second pass: dequantize FP8 tensors in batches
        // Each batch: (1) eval raw mmap'd data on CPU, (2) GPU dequant, (3) eval results
        let batchSize = 4
        for batchStart in stride(from: 0, to: fp8Batch.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, fp8Batch.count)
            let batch = Array(fp8Batch[batchStart..<batchEnd])

            let results = dequantizeFP8Batch(batch.map { ($0.weight, $0.scale) }, batchIndex: batchStart / batchSize)

            for (i, result) in results.enumerated() {
                let key = batch[i].key
                classifyWeight(key: key, weight: result, diffusionPrefix: diffusionPrefix,
                             connectorPrefix: connectorPrefix, vaePrefix: vaePrefix,
                             textEmbedPrefix: textEmbedPrefix,
                             transformer: &transformer, vae: &vae, textEncoder: &textEncoder)
            }

            let done = batchEnd
            if done <= batchSize * 2 || done % (batchSize * 20) == 0 || done == fp8Batch.count {
                let elapsed = Date().timeIntervalSince(dequantStart)
                LTXDebug.log("  FP8 dequant: \(done)/\(fp8Batch.count) done (\(String(format: "%.1f", elapsed))s elapsed)")
            }
        }

        if isFP8 {
            let dequantTime = Date().timeIntervalSince(dequantStart)
            LTXDebug.log("Dequantized \(fp8Batch.count) FP8 weights (of \(fp8Scales.count) scale factors) in \(String(format: "%.1f", dequantTime))s")
        }
        LTXDebug.log("Split unified weights: transformer=\(transformer.count), vae=\(vae.count), textEncoder=\(textEncoder.count)")
        return LTXComponentWeights(
            transformer: transformer,
            vaeDecoder: vae,
            textEncoder: textEncoder
        )
    }

    /// Classify a weight tensor into the appropriate component dictionary
    private static func classifyWeight(
        key: String, weight: MLXArray,
        diffusionPrefix: String, connectorPrefix: String,
        vaePrefix: String, textEmbedPrefix: String,
        transformer: inout [String: MLXArray],
        vae: inout [String: MLXArray],
        textEncoder: inout [String: MLXArray]
    ) {
        if key.hasPrefix(textEmbedPrefix) {
            textEncoder[key] = weight
        } else if key.hasPrefix(connectorPrefix) {
            textEncoder[String(key.dropFirst(diffusionPrefix.count))] = weight
        } else if key.hasPrefix(vaePrefix) {
            vae[String(key.dropFirst(vaePrefix.count))] = weight
        } else if key.hasPrefix(diffusionPrefix) {
            transformer[String(key.dropFirst(diffusionPrefix.count))] = weight
        }
    }

    // MARK: - File Loading

    /// Load all weights from a model directory
    /// - Parameter modelPath: Path to directory containing safetensors files
    /// - Returns: Dictionary of weight name to MLXArray
    public static func loadWeights(from modelPath: String) throws -> [String: MLXArray] {
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
    public static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        try loadWeights(from: url.path)
    }

    /// Load a single safetensors file
    public static func loadSingleFile(path: String) throws -> [String: MLXArray] {
        LTXDebug.log("Loading safetensors from: \(path)")
        let weights = try loadArrays(url: URL(fileURLWithPath: path))
        LTXDebug.log("Loaded \(weights.count) tensors")
        return weights
    }

    /// Load a single safetensors file from URL
    public static func loadSingleFile(url: URL) throws -> [String: MLXArray] {
        try loadSingleFile(path: url.path)
    }

    // MARK: - Weight Mapping

    /// Map Python transformer weight keys to Swift module paths
    ///
    /// The Swift model uses snake_case keys matching @ModuleInfo declarations.
    /// This function maps structural differences between safetensors and Swift model.
    public static func mapTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            if let newKey = mapTransformerKey(key) {
                mapped[newKey] = value
            }
            // Skip keys that return nil (audio-related keys we don't need)
        }

        LTXDebug.log("Mapped \(mapped.count) transformer weights (from \(weights.count) total)")

        // Debug: show sample key mappings
        let sampleOrigKeys = ["proj_in.weight", "time_embed.emb.timestep_embedder.linear_1.weight",
                              "transformer_blocks.0.attn1.norm_q.weight", "transformer_blocks.0.ff.net.0.proj.weight",
                              "transformer_blocks.0.attn1.to_out.0.weight"]
        LTXDebug.log("Sample key mappings:")
        for origKey in sampleOrigKeys {
            if let newKey = mapTransformerKey(origKey) {
                LTXDebug.log("  '\(origKey)' → '\(newKey)'")
            }
        }

        return mapped
    }

    /// Map a single transformer key from safetensors to Swift model format
    ///
    /// Returns nil for keys that should be skipped (e.g., audio-related keys)
    private static func mapTransformerKey(_ key: String) -> String? {
        // Skip audio-related keys (LTX-2 has audio support but we're video-only)
        // This includes:
        // - audio_* top-level keys (audio_proj_in, audio_scale_shift_table, etc.)
        // - .audio_* block keys (audio_attn1, audio_ff, etc.)
        // - av_cross_attn_* (audio-video cross attention)
        // - video_to_audio_* and video_a2v_* (video-to-audio cross attention)
        if key.hasPrefix("audio_") ||
           key.contains(".audio_") ||
           key.hasPrefix("av_cross_attn_") ||
           key.contains("video_to_audio") ||
           key.contains("video_a2v") ||
           key.contains("a2v_ca") ||
           key.contains("scale_shift_table_a2v") {
            return nil
        }

        var k = key

        // Top-level structural mappings (safetensors → Swift @ModuleInfo)
        // The Swift model uses different names for some top-level modules:
        // - proj_in → patchify_proj (input projection)
        // - time_embed.emb.timestep_embedder → adaln_single.emb (timestep embedding)
        k = k.replacingOccurrences(of: "proj_in.", with: "patchify_proj.")
        k = k.replacingOccurrences(of: "proj_out.", with: "proj_out.")  // Same name

        // AdaLN: Remove the extra timestep_embedder level
        // safetensors: time_embed.emb.timestep_embedder.linear_1 → Swift: adaln_single.emb.linear_1
        k = k.replacingOccurrences(of: "time_embed.emb.timestep_embedder.", with: "adaln_single.emb.")
        k = k.replacingOccurrences(of: "time_embed.linear.", with: "adaln_single.linear.")
        // Some safetensors also have adaln_single.emb.timestep_embedder (redundant with time_embed)
        k = k.replacingOccurrences(of: "adaln_single.emb.timestep_embedder.", with: "adaln_single.emb.")

        // Attention norm mappings: safetensors uses norm_q/norm_k, Swift uses q_norm/k_norm
        k = k.replacingOccurrences(of: ".norm_q.", with: ".q_norm.")
        k = k.replacingOccurrences(of: ".norm_k.", with: ".k_norm.")

        // Remove indexed to_out: safetensors uses to_out.0, Swift uses to_out
        k = k.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

        // FFN mappings: safetensors uses ff.net.{0,2}, Swift uses ff.{project_in,project_out}
        k = k.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.project_in.proj.")
        k = k.replacingOccurrences(of: ".ff.net.2.", with: ".ff.project_out.")

        // Caption projection linear layer mapping
        k = k.replacingOccurrences(of: ".linear_1.", with: ".linear_1.")  // Same (already snake_case)
        k = k.replacingOccurrences(of: ".linear_2.", with: ".linear_2.")  // Same (already snake_case)

        return k
    }

    /// Map VAE weight keys from safetensors to Swift module paths
    ///
    /// Maps flat 7 up_blocks structure: `up_blocks.{i}.` → `up_blocks_{i}.`
    /// Maps per-channel statistics: `per_channel_statistics.mean-of-means` → `mean_of_means`
    /// Strips `decoder.` prefix from all keys.
    public static func mapVAEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip encoder weights (we only need decoder for generation)
            if key.hasPrefix("encoder.") { continue }

            // Handle per-channel statistics
            // Note: must use exact suffix matching to distinguish:
            //   "per_channel_statistics.std-of-means" from
            //   "per_channel_statistics.mean-of-stds_over_std-of-means"
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

            // Convert up_blocks.{i}. → up_blocks_{i}. (dots to underscores for block index)
            for i in 0...6 {
                let src = "up_blocks.\(i)."
                if newKey.hasPrefix(src) {
                    newKey = "up_blocks_\(i)." + String(newKey.dropFirst(src.count))
                    break
                }
            }

            mapped[newKey] = value
        }

        LTXDebug.log("Mapped \(mapped.count) VAE decoder weights (encoder weights skipped)")
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
    /// **Format 2 — Unified file** (split by `splitUnifiedWeights`):
    /// - `text_embedding_projection.*` → `feature_extractor.*`
    /// - `video_embeddings_connector.*` → `embeddings_connector.*`
    public static func mapTextEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip audio keys
            if key.contains("audio") { continue }

            var newKey: String? = nil

            // === Format 1: Standalone connector file ===

            if key.hasPrefix("text_proj_in.") {
                // text_proj_in.weight → feature_extractor.aggregate_embed.weight
                newKey = key.replacingOccurrences(of: "text_proj_in.", with: "feature_extractor.aggregate_embed.")
            } else if key.hasPrefix("video_connector.") {
                // video_connector.* → embeddings_connector.* (with internal remapping)
                var k = key.replacingOccurrences(of: "video_connector.", with: "embeddings_connector.")
                k = applyConnectorInternalMapping(k)
                newKey = k
            }

            // === Format 2: Unified file (split by component) ===

            else if key.hasPrefix("text_embedding_projection.") {
                // text_embedding_projection.* → feature_extractor.*
                newKey = key.replacingOccurrences(of: "text_embedding_projection.", with: "feature_extractor.")
            } else if key.hasPrefix("video_embeddings_connector.") {
                // video_embeddings_connector.* → embeddings_connector.* (with internal remapping)
                var k = key.replacingOccurrences(of: "video_embeddings_connector.", with: "embeddings_connector.")
                k = applyConnectorInternalMapping(k)
                newKey = k
            }

            if let newKey = newKey {
                mapped[newKey] = value
            } else {
                LTXDebug.log("TextEncoder: Skipping unknown key: \(key)")
            }
        }

        LTXDebug.log("Mapped \(mapped.count) text encoder weights")
        return mapped
    }

    /// Apply internal key remapping for connector transformer blocks
    ///
    /// Shared between standalone and unified formats.
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

    /// Apply weights to a transformer model
    public static func applyTransformerWeights(
        _ weights: [String: MLXArray],
        to model: LTXTransformer
    ) throws {
        let mapped = mapTransformerWeights(weights)

        // Flatten model parameters for key matching
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        // Debug: show model parameter keys structure
        LTXDebug.log("Model has \(flatParameters.count) parameter keys")
        let sampleModelKeys = flatParameters.keys.sorted().prefix(20)
        LTXDebug.log("Sample model parameter keys:")
        for key in sampleModelKeys {
            LTXDebug.log("  \(key)")
        }

        var updates: [String: MLXArray] = [:]
        var notFound = 0
        var notFoundKeys: [String] = []

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 10 {
                    notFoundKeys.append(key)
                }
            }
        }

        if !notFoundKeys.isEmpty {
            LTXDebug.log("Missing parameter keys (first 10):")
            for key in notFoundKeys {
                LTXDebug.log("  \(key)")
            }
        }
        if notFound > 10 {
            LTXDebug.log("... and \(notFound - 10) more missing parameters")
        }

        // Update model with new weights
        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        LTXDebug.log("Applied \(updates.count) weights to transformer (\(notFound) not found)")
    }

    /// Apply weights to a VAE decoder model
    public static func applyVAEWeights(
        _ weights: [String: MLXArray],
        to model: VideoDecoder
    ) throws {
        let mapped = mapVAEWeights(weights)

        // Flatten model parameters for key matching
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        var updates: [String: MLXArray] = [:]
        var notFound = 0

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 10 {
                    LTXDebug.log("VAE Warning: No parameter found for key: \(key)")
                }
            }
        }

        if notFound > 10 {
            LTXDebug.log("... and \(notFound - 10) more missing VAE parameters")
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        LTXDebug.log("Applied \(updates.count) weights to VAE (\(notFound) not found)")
    }

    /// Apply weights to a text encoder model
    public static func applyTextEncoderWeights(
        _ weights: [String: MLXArray],
        to model: VideoGemmaTextEncoderModel
    ) throws {
        let mapped = mapTextEncoderWeights(weights)

        // Flatten model parameters for key matching
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        var updates: [String: MLXArray] = [:]
        var notFound = 0

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 10 {
                    LTXDebug.log("TextEncoder Warning: No parameter found for key: \(key)")
                }
            }
        }

        if notFound > 10 {
            LTXDebug.log("... and \(notFound - 10) more missing TextEncoder parameters")
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        LTXDebug.log("Applied \(updates.count) weights to TextEncoder (\(notFound) not found)")
    }

    /// Get summary of loaded weights
    public static func summarizeWeights(_ weights: [String: MLXArray]) {
        var totalParams: Int64 = 0
        var byPrefix: [String: Int64] = [:]

        for (key, array) in weights {
            let params = Int64(array.shape.reduce(1, *))
            totalParams += params

            // Group by first component
            let prefix = String(key.split(separator: ".").first ?? Substring(key))
            byPrefix[prefix, default: 0] += params
        }

        LTXDebug.log("Weight Summary:")
        for (prefix, params) in byPrefix.sorted(by: { $0.value > $1.value }) {
            let gb = Float(params * 2) / 1_000_000_000  // bf16 = 2 bytes
            LTXDebug.log("  \(prefix): \(params) params (~\(String(format: "%.2f", gb))GB)")
        }
        LTXDebug.log("Total: \(totalParams) parameters")
    }
}

// Need to import MLX for loadArrays function
@preconcurrency import MLX
import MLXNN
