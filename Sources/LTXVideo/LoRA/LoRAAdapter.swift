// LoRAAdapter.swift - Apply LoRA to LTX-2 Transformer
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - LoRA Adapter

/// Applies LoRA adaptations to the LTX transformer model
class LoRAAdapter {
    /// The loaded LoRA weights
    let loraWeights: LoRAWeights

    /// Whether to fuse weights (permanent merge) vs runtime application
    let fused: Bool

    init(loraWeights: LoRAWeights, fused: Bool = true) {
        self.loraWeights = loraWeights
        self.fused = fused
    }

    /// Count layers that would be modified by this LoRA on a transformer
    ///
    /// - Parameters:
    ///   - model: The transformer model to check
    /// - Returns: Number of layers that would be modified
    @discardableResult
    func applyFused(to model: LTXTransformer) -> Int {
        var modifiedCount = 0

        // Count matching layers for each transformer block.
        // LoRA originalKeys use the 'diffusion_model.' prefix, so we include it here.
        for blockIdx in 0..<model.transformerBlocks.count {
            let blockPrefix = "diffusion_model.transformer_blocks.\(blockIdx)"

            // Attention layers
            modifiedCount += countMatchingLayers(prefix: "\(blockPrefix).attn1")
            modifiedCount += countMatchingLayers(prefix: "\(blockPrefix).attn2")

            // Feed-forward layers
            modifiedCount += countMatchingFFLayers(prefix: "\(blockPrefix).ff")
        }

        LTXDebug.log("LoRA would modify \(modifiedCount) layers")
        return modifiedCount
    }

    /// Fuse LoRA weights into the transformer model's weights
    ///
    /// Formula: W' = W + scale * (B @ A)
    ///
    /// Uses batched processing (grouping layers by transformer block) to reduce peak memory.
    /// Pattern from flux-2-swift-mlx: apply model.update() + eval() per batch instead of per layer.
    /// This reduces update calls from ~576 to ~48 and improves GPU utilization.
    ///
    /// Handles quantized models: if the transformer has been quantized (QuantizedLinear layers),
    /// uses dequant → merge → requant pattern to preserve quantization after LoRA fusion.
    ///
    /// - Parameters:
    ///   - model: The transformer to fuse into
    /// - Returns: Dictionary of original weights (for unfusing later)
    @discardableResult
    func fuseWeights(into model: Module) -> [String: MLXArray] {
        var originalWeights: [String: MLXArray] = [:]
        var fusedCount = 0

        // Get model parameters as a flat dictionary (ONCE — avoids repeated flattening)
        let modelParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        // Build a map of quantized modules for dequant→merge→requant
        var quantizedModules: [String: QuantizedLinear] = [:]
        for (path, module) in model.leafModules().flattened() {
            if let ql = module as? QuantizedLinear {
                quantizedModules[path] = ql
            }
        }
        let isQuantized = !quantizedModules.isEmpty
        if isQuantized {
            LTXDebug.log("LoRA: Detected quantized model (\(quantizedModules.count) QuantizedLinear layers)")
        }

        // Group LoRA layers by transformer block for batched updates.
        // This reduces peak memory from ~model_size to ~block_size (~98% reduction)
        // by applying model.update() + eval() per batch instead of accumulating all updates.
        let batches = Self.groupLayersByBlock(loraWeights.layers)
        LTXDebug.log("LoRA: Processing \(loraWeights.layers.count) layers in \(batches.count) batches")

        for (batchPrefix, layers) in batches {
            var batchUpdates: [(String, MLXArray)] = []

            for layer in layers {
                guard let delta = loraWeights.getDelta(for: layer.originalKey) else { continue }

                // Translate LoRA key (ComfyUI/Diffusers format) to Swift model flattened key.
                let swiftKey = LoRAKeyMapper.loraKeyToModelKey(layer.originalKey)

                // Derive the module path (strip ".weight" suffix)
                let layerPath = swiftKey.hasSuffix(".weight")
                    ? String(swiftKey.dropLast(".weight".count))
                    : swiftKey

                // Check if this layer is a QuantizedLinear module
                if let ql = quantizedModules[layerPath] {
                    // === QuantizedLinear path: dequant → merge → requant ===

                    // 1. Dequantize MLX qint → float16
                    var mergedWeight = dequantized(
                        ql.weight, scales: ql.scales, biases: ql.biases,
                        groupSize: ql.groupSize, bits: ql.bits
                    ).asType(.float16)

                    // Save original quantized state for unfusing
                    originalWeights[swiftKey] = ql.weight
                    originalWeights[layerPath + ".scales"] = ql.scales
                    if let b = ql.biases { originalWeights[layerPath + ".biases"] = b }

                    // 2. Apply LoRA delta
                    let deltaConverted = delta.asType(mergedWeight.dtype)
                    mergedWeight = mergedWeight + deltaConverted

                    // 3. Requantize
                    let (newWeight, newScales, newBiases) = quantized(
                        mergedWeight, groupSize: ql.groupSize, bits: ql.bits
                    )

                    // 4. Update weight, scales, and biases
                    batchUpdates.append((swiftKey, newWeight))
                    batchUpdates.append((layerPath + ".scales", newScales))
                    if let nb = newBiases {
                        batchUpdates.append((layerPath + ".biases", nb))
                    }
                    fusedCount += 1
                } else {
                    // === Standard Linear path ===
                    guard let currentWeight = modelParams[swiftKey] else {
                        LTXDebug.log("LoRA fuse: no model weight for \(swiftKey), skipping")
                        continue
                    }

                    // Save original for unfusing
                    originalWeights[swiftKey] = currentWeight

                    // Fuse: W' = W + delta (delta already includes scale)
                    let deltaConverted = delta.asType(currentWeight.dtype)
                    let newWeight = currentWeight + deltaConverted
                    batchUpdates.append((swiftKey, newWeight))
                    fusedCount += 1
                }
            }

            // Apply this batch and materialize to free intermediate arrays
            if !batchUpdates.isEmpty {
                let params = ModuleParameters.unflattened(batchUpdates)
                model.update(parameters: params)
                eval(model.parameters())
                LTXDebug.log("  Batch '\(batchPrefix)': fused \(batchUpdates.count) layers")
            }
        }

        // Clear GPU cache after all batches (matching flux-2 pattern)
        Memory.clearCache()

        LTXDebug.log("LoRA fused \(fusedCount) layers (saved \(originalWeights.count) originals)")
        return originalWeights
    }

    /// Group LoRA layers by transformer block prefix for batched processing
    ///
    /// Matching flux-2-swift-mlx pattern: group by "diffusion_model.transformer_blocks.N"
    /// so all layers in one block (attn1.to_q, attn1.to_k, ..., ff.net.0.proj, etc.)
    /// are fused in a single model.update() + eval() call.
    private static func groupLayersByBlock(_ layers: [LoRALayerInfo]) -> [(String, [LoRALayerInfo])] {
        var groups: [(String, [LoRALayerInfo])] = []
        var prefixIndex: [String: Int] = [:]

        for layer in layers {
            let prefix = blockPrefix(for: layer.originalKey)
            if let idx = prefixIndex[prefix] {
                groups[idx].1.append(layer)
            } else {
                prefixIndex[prefix] = groups.count
                groups.append((prefix, [layer]))
            }
        }
        return groups
    }

    /// Extract transformer block prefix from a LoRA key
    ///
    /// "diffusion_model.transformer_blocks.5.attn1.to_q" → "transformer_blocks.5"
    private static func blockPrefix(for layerPath: String) -> String {
        let components = layerPath.split(separator: ".")

        // Strip "diffusion_model." prefix if present
        let start: Int
        if components.first == "diffusion_model" {
            start = 1
        } else {
            start = 0
        }

        // Match "transformer_blocks.N"
        if components.count > start + 1,
           components[start] == "transformer_blocks",
           Int(components[start + 1]) != nil
        {
            return "\(components[start]).\(components[start + 1])"
        }

        return String(components.first ?? Substring(layerPath))
    }

    /// Unfuse previously fused LoRA weights by restoring originals
    ///
    /// - Parameters:
    ///   - originalWeights: Dictionary returned by `fuseWeights(into:)`
    ///   - model: The transformer to restore
    static func unfuseWeights(from originalWeights: [String: MLXArray], into model: Module) {
        var restoredCount = 0
        for (keyPath, originalWeight) in originalWeights {
            setParameterByPath(model: model, keyPath: keyPath, value: originalWeight)
            restoredCount += 1
        }
        MLX.eval(model.parameters())
        LTXDebug.log("LoRA unfused: restored \(restoredCount) layers")
    }

    /// Count layers that would be modified by this LoRA
    private func countMatchingLayers(prefix: String) -> Int {
        var count = 0

        // Check for Q, K, V, out projections
        for suffix in ["to_q", "to_k", "to_v", "to_out.0"] {
            if loraWeights.hasLayer("\(prefix).\(suffix)") {
                count += 1
            }
        }

        return count
    }

    /// Count feed-forward layers that would be modified
    private func countMatchingFFLayers(prefix: String) -> Int {
        var count = 0

        // Check for input and output projections
        if loraWeights.hasLayer("\(prefix).net.0.proj") {
            count += 1
        }
        if loraWeights.hasLayer("\(prefix).net.2") {
            count += 1
        }

        return count
    }

    /// Compute LoRA deltas for all layers
    /// Returns dictionary of layer key -> delta tensor
    func computeDeltas() -> [String: MLXArray] {
        var deltas: [String: MLXArray] = [:]

        for layer in loraWeights.layers {
            if let delta = loraWeights.getDelta(for: layer.originalKey) {
                deltas[layer.originalKey] = delta
            }
        }

        return deltas
    }

    /// Get scheduler overrides if this LoRA specifies them
    var schedulerOverrides: LoRASchedulerOverrides? {
        // This would need to be extracted from LoRA metadata
        // For now, return nil - can be enhanced based on LoRA format
        return nil
    }
}

// MARK: - LoRA Application Result

/// Result of applying LoRA to a model
public struct LoRAApplicationResult: Sendable {
    /// Number of layers modified
    public let modifiedLayerCount: Int

    /// Name of the applied LoRA
    public let loraName: String

    /// Scale factor used
    public let scale: Float

    /// Whether weights were fused
    public let fused: Bool

    /// Any scheduler overrides to apply
    public let schedulerOverrides: LoRASchedulerOverrides?
}

// MARK: - Multi-LoRA Support

/// Applies multiple LoRAs to a model
class MultiLoRAAdapter {
    /// Individual adapters
    let adapters: [LoRAAdapter]

    /// Composition mode
    let compositionMode: LoRACompositionMode

    init(
        loraConfigs: [LoRAConfig],
        compositionMode: LoRACompositionMode = .add
    ) throws {
        self.compositionMode = compositionMode

        self.adapters = try loraConfigs.map { config in
            let weights = try LoRALoader.load(from: config.weightsPath, config: config)
            return LoRAAdapter(loraWeights: weights, fused: config.fused)
        }
    }

    /// Apply all LoRAs to the model
    ///
    /// For `.add` mode: W' = W + sum(scale_i * delta_i)
    /// For `.multiply` mode: Applies sequentially
    @discardableResult
    func apply(to model: LTXTransformer) -> [LoRAApplicationResult] {
        var results: [LoRAApplicationResult] = []

        for adapter in adapters {
            let count = adapter.applyFused(to: model)
            results.append(LoRAApplicationResult(
                modifiedLayerCount: count,
                loraName: adapter.loraWeights.info.name,
                scale: adapter.loraWeights.scale,
                fused: adapter.fused,
                schedulerOverrides: adapter.schedulerOverrides
            ))
        }

        return results
    }

    /// Get combined scheduler overrides (last non-nil wins)
    var schedulerOverrides: LoRASchedulerOverrides? {
        for adapter in adapters.reversed() {
            if let overrides = adapter.schedulerOverrides {
                return overrides
            }
        }
        return nil
    }
}

// MARK: - Convenience Extensions

extension LTXTransformer {
    /// Apply a LoRA to this transformer (count only — for compatibility)
    ///
    /// - Parameters:
    ///   - loraPath: Path to the LoRA .safetensors file
    ///   - scale: Scale factor (default: 1.0)
    /// - Returns: Application result
    @discardableResult
    func applyLoRA(
        from loraPath: String,
        scale: Float = 1.0
    ) throws -> LoRAApplicationResult {
        let config = LoRAConfig(weightsPath: loraPath, scale: scale)
        let weights = try LoRALoader.load(from: loraPath, config: config)
        let adapter = LoRAAdapter(loraWeights: weights, fused: true)

        let count = adapter.applyFused(to: self)

        return LoRAApplicationResult(
            modifiedLayerCount: count,
            loraName: weights.info.name,
            scale: scale,
            fused: true,
            schedulerOverrides: adapter.schedulerOverrides
        )
    }
}

extension Module {
    /// Apply a LoRA with real weight fusion
    ///
    /// Works with both LTXTransformer (video-only) and LTX2Transformer (video+audio).
    ///
    /// - Parameters:
    ///   - loraPath: Path to the LoRA .safetensors file
    ///   - scale: Scale factor (default: 1.0)
    /// - Returns: Dictionary of original weights (for unfusing), and application result
    @discardableResult
    func fuseLoRA(
        from loraPath: String,
        scale: Float = 1.0
    ) throws -> (originalWeights: [String: MLXArray], result: LoRAApplicationResult) {
        let config = LoRAConfig(weightsPath: loraPath, scale: scale)
        let weights = try LoRALoader.load(from: loraPath, config: config)
        let adapter = LoRAAdapter(loraWeights: weights, fused: true)

        let originals = adapter.fuseWeights(into: self)

        let result = LoRAApplicationResult(
            modifiedLayerCount: originals.count,
            loraName: weights.info.name,
            scale: scale,
            fused: true,
            schedulerOverrides: adapter.schedulerOverrides
        )

        return (originals, result)
    }

    /// Unfuse previously fused LoRA weights
    func unfuseLoRA(originalWeights: [String: MLXArray]) {
        LoRAAdapter.unfuseWeights(from: originalWeights, into: self)
    }
}

// MARK: - Parameter Path Helper

/// Set a parameter on a Module by dot-separated key path
/// Navigates the Module tree using key components to find and update the target parameter.
private func setParameterByPath(model: Module, keyPath: String, value: MLXArray) {
    let components = keyPath.split(separator: ".").map(String.init)
    guard !components.isEmpty else { return }

    // Build a NestedDictionary update for Module.update()
    // We need to build: {"component1": {"component2": {"leaf": value}}}
    // Use NestedDictionary.unflattened to build from the flat key path
    let flatKey = components.joined(separator: ".")
    let params = ModuleParameters.unflattened([(flatKey, value)])
    model.update(parameters: params)
}
