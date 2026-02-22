// LoRALoader.swift - LoRA Weight Loading for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - LoRA Loader

/// Loads LoRA weights from SafeTensors files
class LoRALoader {
    /// Load LoRA weights from a file
    ///
    /// - Parameters:
    ///   - path: Path to the .safetensors file
    ///   - config: Optional LoRA configuration
    /// - Returns: Dictionary of LoRA weight arrays and metadata
    static func load(
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
struct LoRAWeights: Sendable {
    /// Raw weight arrays keyed by parameter name
    let weights: [String: MLXArray]

    /// Parsed layer information
    let layers: [LoRALayerInfo]

    /// Metadata about the LoRA
    let info: LoRAInfo

    /// Scale factor to apply
    let scale: Float

    /// Get LoRA delta for a specific layer
    ///
    /// Computes: scale * (B @ A) where A is down and B is up
    ///
    /// - Parameters:
    ///   - layerKey: The original layer key to get delta for
    /// - Returns: The LoRA delta matrix, or nil if not found
    func getDelta(for layerKey: String) -> MLXArray? {
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
    func hasLayer(_ layerKey: String) -> Bool {
        return layers.contains { $0.originalKey == layerKey }
    }
}

// MARK: - Weight Key Mapping

/// Maps between LoRA file key format and Swift model parameter key format
struct LoRAKeyMapper {
    /// Translate a LoRA originalKey to the flattened model parameter key used by
    /// MLX Module.parameters().flattened().
    ///
    /// LoRA files (ComfyUI / Diffusers format) use keys like:
    ///   diffusion_model.transformer_blocks.0.attn1.to_out.0
    ///   diffusion_model.transformer_blocks.0.ff.net.0.proj
    ///   diffusion_model.transformer_blocks.0.ff.net.2
    ///
    /// The Swift model exposes flattened keys like:
    ///   transformer_blocks.0.attn1.to_out.weight
    ///   transformer_blocks.0.ff.project_in.proj.weight
    ///   transformer_blocks.0.ff.project_out.weight
    ///
    /// Transformations applied:
    ///   1. Strip leading "diffusion_model." prefix
    ///   2. Replace ".to_out.0" with ".to_out"  (list index removed; module key is just "to_out")
    ///   3. Replace ".ff.net.0.proj" with ".ff.project_in.proj"  (LTXFeedForward key "project_in")
    ///   4. Replace ".ff.net.2" with ".ff.project_out"  (LTXFeedForward key "project_out")
    ///   5. Append ".weight" (Linear weight parameter)
    static func loraKeyToModelKey(_ loraOriginalKey: String) -> String {
        var key = loraOriginalKey

        // 1. Strip the "diffusion_model." wrapper prefix used by ComfyUI/Diffusers LoRA files
        if key.hasPrefix("diffusion_model.") {
            key = String(key.dropFirst("diffusion_model.".count))
        }

        // 2. to_out.0 -> to_out
        //    The Python Sequential wrapper adds the ".0" list index; the Swift module
        //    LTXAttention declares @ModuleInfo(key: "to_out") with no index.
        key = key.replacingOccurrences(of: ".to_out.0", with: ".to_out")

        // 3. ff.net.0.proj -> ff.project_in.proj
        //    Python: FeedForward contains nn.Sequential("net") with [GEGLU(.proj), ..., Linear]
        //    Swift: LTXFeedForward uses @ModuleInfo(key: "project_in") var projectIn: GELUApprox
        //           and GELUApprox uses @ModuleInfo var proj: Linear  (default key "proj")
        key = key.replacingOccurrences(of: ".ff.net.0.proj", with: ".ff.project_in.proj")

        // 4. ff.net.2 -> ff.project_out
        //    Python: FeedForward's net[2] is the output Linear
        //    Swift: LTXFeedForward uses @ModuleInfo(key: "project_out") var projectOut: Linear
        key = key.replacingOccurrences(of: ".ff.net.2", with: ".ff.project_out")

        // 5. Append .weight — all targeted layers are Linear modules whose weight
        //    parameter is named "weight"
        key = key + ".weight"

        return key
    }

    /// Legacy camelCase mapper (not used for LoRA fusion, kept for reference)
    static func mapKey(_ pythonKey: String) -> String {
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
    static func possibleSwiftKeys(for pythonKey: String) -> [String] {
        return [
            mapKey(pythonKey),
            pythonKey  // Also try unchanged
        ]
    }
}
