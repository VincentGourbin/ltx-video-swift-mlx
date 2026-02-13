// LTXModelRegistry.swift - Model Registry and Paths
// Copyright 2025

import Foundation

/// Registry for LTX-2 model components and paths
public enum LTXModelRegistry {
    // MARK: - Model Components

    /// Model component types
    public enum ModelComponent: Equatable, Sendable {
        case transformer(LTXModel)
        case vaeDecoder
        case textEncoder

        public var displayName: String {
            switch self {
            case .transformer(let model):
                return "Transformer (\(model.displayName))"
            case .vaeDecoder:
                return "VAE Decoder"
            case .textEncoder:
                return "Text Encoder (Gemma3)"
            }
        }
    }

    // MARK: - Directories

    /// Base directory for cached models
    public static var modelsDirectory: URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("ltx-video-mlx", isDirectory: true)
    }

    /// Directory for transformer weights
    public static func transformerDirectory(for model: LTXModel) -> URL {
        modelsDirectory.appendingPathComponent("transformer-\(model.rawValue)", isDirectory: true)
    }

    /// Directory for VAE weights
    public static var vaeDirectory: URL {
        modelsDirectory.appendingPathComponent("vae", isDirectory: true)
    }

    /// Directory for text encoder weights
    public static var textEncoderDirectory: URL {
        modelsDirectory.appendingPathComponent("text-encoder", isDirectory: true)
    }

    // MARK: - HuggingFace Repositories

    /// HuggingFace repository for the given model
    public static func huggingFaceRepo(for model: LTXModel) -> String {
        return model.huggingFaceRepo
    }

    /// HuggingFace repository for Gemma3 text encoder
    public static var textEncoderRepo: String {
        "Acelogic/ltx-video-2b-v0.9.7-mlx"  // Text encoder is bundled with the model
    }

    // MARK: - Files

    /// Expected files for transformer
    public static let transformerFiles: [String] = [
        "transformer.safetensors",
        "config.json"
    ]

    /// Expected files for VAE decoder
    public static let vaeFiles: [String] = [
        "vae_decoder.safetensors",
        "vae_config.json"
    ]

    /// Expected files for text encoder
    public static let textEncoderFiles: [String] = [
        "text_encoder.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    // MARK: - Status Checking

    /// Check if a component is downloaded
    public static func isDownloaded(_ component: ModelComponent) -> Bool {
        let dir: URL
        let files: [String]

        switch component {
        case .transformer(let model):
            dir = transformerDirectory(for: model)
            files = transformerFiles
        case .vaeDecoder:
            dir = vaeDirectory
            files = vaeFiles
        case .textEncoder:
            dir = textEncoderDirectory
            files = textEncoderFiles
        }

        // Check if all required files exist
        return files.allSatisfy { file in
            FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path)
        }
    }

    /// Get missing files for a component
    public static func missingFiles(for component: ModelComponent) -> [String] {
        let dir: URL
        let files: [String]

        switch component {
        case .transformer(let model):
            dir = transformerDirectory(for: model)
            files = transformerFiles
        case .vaeDecoder:
            dir = vaeDirectory
            files = vaeFiles
        case .textEncoder:
            dir = textEncoderDirectory
            files = textEncoderFiles
        }

        return files.filter { file in
            !FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path)
        }
    }

    // MARK: - System Info

    /// System RAM in GB
    public static var systemRAMGB: Int {
        let bytes = ProcessInfo.processInfo.physicalMemory
        return Int(bytes / 1_073_741_824)  // Convert to GB
    }

    /// Recommended model based on available RAM
    public static var recommendedModel: LTXModel {
        let ram = systemRAMGB
        if ram >= 32 {
            return .dev
        } else if ram >= 18 {
            return .distilled
        } else {
            return .distilledFP8
        }
    }
}
