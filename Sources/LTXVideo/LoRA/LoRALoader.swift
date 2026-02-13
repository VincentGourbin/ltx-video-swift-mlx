// LoRALoader.swift - LoRA Weight Loading for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - LoRA Loader

/// Loads LoRA weights from SafeTensors files
public class LoRALoader {
    /// Load LoRA weights from a file
    ///
    /// - Parameters:
    ///   - path: Path to the .safetensors file
    ///   - config: Optional LoRA configuration
    /// - Returns: Dictionary of LoRA weight arrays and metadata
    public static func load(
        from path: String,
        config: LoRAConfig? = nil
    ) throws -> LoRAWeights {
        let url = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: path) else {
            throw LTXError.fileNotFound(path)
        }

        // Load weights from SafeTensors
        let weights = try loadSafeTensors(from: url)

        // Parse LoRA structure
        let loraLayers = parseLoRALayers(from: weights)

        // Get metadata
        let rank = inferRank(from: weights)
        let targetModules = inferTargetModules(from: weights)

        let info = LoRAInfo(
            name: config?.name ?? url.deletingPathExtension().lastPathComponent,
            path: path,
            layerCount: loraLayers.count,
            rank: rank,
            targetModules: targetModules,
            hasSchedulerOverrides: config?.schedulerOverrides != nil,
            fileSizeBytes: try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64
        )

        return LoRAWeights(
            weights: weights,
            layers: loraLayers,
            info: info,
            scale: config?.scale ?? 1.0
        )
    }

    /// Load SafeTensors file
    private static func loadSafeTensors(from url: URL) throws -> [String: MLXArray] {
        // Use MLX's built-in safetensors loading
        return try MLX.loadArrays(url: url)
    }

    /// Parse LoRA layer structure from weight keys
    private static func parseLoRALayers(from weights: [String: MLXArray]) -> [LoRALayerInfo] {
        var layers: [LoRALayerInfo] = []
        var processedKeys = Set<String>()

        for key in weights.keys {
            // Look for lora_down patterns (the "A" matrix)
            guard key.contains("lora_down") || key.contains("lora_A") else { continue }
            guard !processedKeys.contains(key) else { continue }

            // Find corresponding up key
            let upKey: String
            let originalKey: String

            if key.contains("lora_down") {
                upKey = key.replacingOccurrences(of: "lora_down", with: "lora_up")
                originalKey = key
                    .replacingOccurrences(of: ".lora_down.weight", with: "")
                    .replacingOccurrences(of: ".lora_down", with: "")
            } else {
                upKey = key.replacingOccurrences(of: "lora_A", with: "lora_B")
                originalKey = key
                    .replacingOccurrences(of: ".lora_A.weight", with: "")
                    .replacingOccurrences(of: ".lora_A", with: "")
            }

            guard weights[upKey] != nil else { continue }

            // Infer rank from down matrix shape
            let downWeight = weights[key]!
            let rank = downWeight.dim(0)

            // Check for alpha
            let alphaKey = originalKey + ".alpha"
            let alpha: Float? = weights[alphaKey].map { Float($0.item(Float.self)) }

            layers.append(LoRALayerInfo(
                originalKey: originalKey,
                downKey: key,
                upKey: upKey,
                rank: rank,
                alpha: alpha
            ))

            processedKeys.insert(key)
            processedKeys.insert(upKey)
        }

        return layers
    }

    /// Infer rank from weights
    private static func inferRank(from weights: [String: MLXArray]) -> Int {
        for (key, value) in weights {
            if key.contains("lora_down") || key.contains("lora_A") {
                return value.dim(0)
            }
        }
        return 0
    }

    /// Infer target modules from weights
    private static func inferTargetModules(from weights: [String: MLXArray]) -> [String] {
        var modules = Set<String>()

        for key in weights.keys {
            for target in LoRATargetType.allCases {
                if key.contains(target.rawValue) {
                    modules.insert(target.rawValue)
                }
            }
        }

        return Array(modules).sorted()
    }
}

// MARK: - LoRA Weights Container

/// Container for loaded LoRA weights
public struct LoRAWeights: Sendable {
    /// Raw weight arrays keyed by parameter name
    public let weights: [String: MLXArray]

    /// Parsed layer information
    public let layers: [LoRALayerInfo]

    /// Metadata about the LoRA
    public let info: LoRAInfo

    /// Scale factor to apply
    public let scale: Float

    /// Get LoRA delta for a specific layer
    ///
    /// Computes: scale * (B @ A) where A is down and B is up
    ///
    /// - Parameters:
    ///   - layerKey: The original layer key to get delta for
    /// - Returns: The LoRA delta matrix, or nil if not found
    public func getDelta(for layerKey: String) -> MLXArray? {
        guard let layer = layers.first(where: { $0.originalKey == layerKey }) else {
            return nil
        }

        guard let down = weights[layer.downKey],
              let up = weights[layer.upKey] else {
            return nil
        }

        // Compute delta: B @ A (up @ down)
        // LoRA decomposition: W' = W + scale * B @ A
        let effectiveScale = scale * layer.effectiveScale
        let delta = MLX.matmul(up, down) * effectiveScale

        return delta
    }

    /// Check if this LoRA has weights for a given layer
    public func hasLayer(_ layerKey: String) -> Bool {
        return layers.contains { $0.originalKey == layerKey }
    }
}

// MARK: - Weight Key Mapping

/// Maps between Python and Swift weight key formats
public struct LoRAKeyMapper {
    /// Common key transformations for LTX-2
    public static func mapKey(_ pythonKey: String) -> String {
        var key = pythonKey

        // Common transformations
        key = key.replacingOccurrences(of: "transformer_blocks", with: "transformerBlocks")
        key = key.replacingOccurrences(of: "to_q", with: "toQ")
        key = key.replacingOccurrences(of: "to_k", with: "toK")
        key = key.replacingOccurrences(of: "to_v", with: "toV")
        key = key.replacingOccurrences(of: "to_out.0", with: "toOut")
        key = key.replacingOccurrences(of: "ff.net.0.proj", with: "ff.projectIn.proj")
        key = key.replacingOccurrences(of: "ff.net.2", with: "ff.projectOut")

        return key
    }

    /// Get all possible Swift keys for a Python key
    public static func possibleSwiftKeys(for pythonKey: String) -> [String] {
        return [
            mapKey(pythonKey),
            pythonKey  // Also try unchanged
        ]
    }
}
