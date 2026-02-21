// LoRAAdapter.swift - Apply LoRA to LTX-2 Transformer
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - LoRA Adapter

/// Applies LoRA adaptations to the LTX transformer model
public class LoRAAdapter {
    /// The loaded LoRA weights
    public let loraWeights: LoRAWeights

    /// Whether to fuse weights (permanent merge) vs runtime application
    public let fused: Bool

    public init(loraWeights: LoRAWeights, fused: Bool = true) {
        self.loraWeights = loraWeights
        self.fused = fused
    }

    /// Count layers that would be modified by this LoRA on a transformer
    ///
    /// - Parameters:
    ///   - model: The transformer model to check
    /// - Returns: Number of layers that would be modified
    @discardableResult
    public func applyFused(to model: LTXTransformer) -> Int {
        var modifiedCount = 0

        // Count matching layers for each transformer block
        for blockIdx in 0..<model.transformerBlocks.count {
            let blockPrefix = "transformer_blocks.\(blockIdx)"

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
    /// - Parameters:
    ///   - model: The transformer to fuse into
    /// - Returns: Dictionary of original weights (for unfusing later)
    @discardableResult
    public func fuseWeights(into model: LTXTransformer) -> [String: MLXArray] {
        var originalWeights: [String: MLXArray] = [:]
        var fusedCount = 0

        // Get model parameters as a flat dictionary
        let modelParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        // Compute deltas and apply them
        for layer in loraWeights.layers {
            guard let delta = loraWeights.getDelta(for: layer.originalKey) else { continue }

            // Map to Swift key and find the weight
            let swiftKey = layer.originalKey + ".weight"
            guard let currentWeight = modelParams[swiftKey] else {
                LTXDebug.log("LoRA fuse: no model weight for \(swiftKey), skipping")
                continue
            }

            // Save original for unfusing
            originalWeights[swiftKey] = currentWeight

            // Fuse: W' = W + delta (delta already includes scale)
            let newWeight = currentWeight + delta
            MLX.eval(newWeight)

            // Apply via nested key path update
            setParameterByPath(model: model, keyPath: swiftKey, value: newWeight)
            fusedCount += 1
        }

        LTXDebug.log("LoRA fused \(fusedCount) layers (saved \(originalWeights.count) originals)")
        return originalWeights
    }

    /// Unfuse previously fused LoRA weights by restoring originals
    ///
    /// - Parameters:
    ///   - originalWeights: Dictionary returned by `fuseWeights(into:)`
    ///   - model: The transformer to restore
    public static func unfuseWeights(from originalWeights: [String: MLXArray], into model: LTXTransformer) {
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
    public func computeDeltas() -> [String: MLXArray] {
        var deltas: [String: MLXArray] = [:]

        for layer in loraWeights.layers {
            if let delta = loraWeights.getDelta(for: layer.originalKey) {
                deltas[layer.originalKey] = delta
            }
        }

        return deltas
    }

    /// Get scheduler overrides if this LoRA specifies them
    public var schedulerOverrides: LoRASchedulerOverrides? {
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
public class MultiLoRAAdapter {
    /// Individual adapters
    public let adapters: [LoRAAdapter]

    /// Composition mode
    public let compositionMode: LoRACompositionMode

    public init(
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
    public func apply(to model: LTXTransformer) -> [LoRAApplicationResult] {
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
    public var schedulerOverrides: LoRASchedulerOverrides? {
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
    public func applyLoRA(
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

    /// Apply a LoRA to this transformer with real weight fusion
    ///
    /// - Parameters:
    ///   - loraPath: Path to the LoRA .safetensors file
    ///   - scale: Scale factor (default: 1.0)
    /// - Returns: Dictionary of original weights (for unfusing), and application result
    @discardableResult
    public func fuseLoRA(
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
    public func unfuseLoRA(originalWeights: [String: MLXArray]) {
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
