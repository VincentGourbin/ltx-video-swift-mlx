// Gemma3WeightLoader.swift - Weight loading for Gemma 3 models
// Follows the MLXLLM loadWeights pattern: sanitize → quantize → update

import Foundation
import MLX
import MLXNN

/// Weight loader for Gemma 3 models (standard and quantized)
enum Gemma3WeightLoader {

    /// Load Gemma 3 weights from a model directory.
    ///
    /// Handles:
    /// - Loading all safetensors files
    /// - Sanitizing keys (language_model prefix, lm_head tying)
    /// - Detecting and applying quantization (4-bit, 8-bit)
    /// - Updating model parameters
    ///
    /// - Parameters:
    ///   - model: The Gemma3TextModel to load weights into
    ///   - weightsDir: Directory containing safetensors + config.json
    static func loadWeights(
        into model: Gemma3TextModel,
        from weightsDir: URL
    ) throws {
        // Step 1: Load all safetensors
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: weightsDir, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
            }
        }

        guard !weights.isEmpty else {
            throw LTXError.weightLoadingFailed("No safetensors files found in \(weightsDir.path)")
        }

        print("[Gemma3] Loaded \(weights.count) weight tensors from safetensors")

        // Step 2: Sanitize keys
        weights = model.sanitize(weights: weights)

        // Step 3: Detect quantization and apply if needed
        if let quantConfig = model.config.quantization {
            print("[Gemma3] Applying quantization: \(quantConfig.bits)-bit, group_size=\(quantConfig.groupSize)")
            quantize(model: model) { path, module in
                // Only quantize layers that have .scales in the weights
                if weights["\(path).scales"] != nil {
                    return (groupSize: quantConfig.groupSize, bits: quantConfig.bits, mode: .affine)
                }
                return nil
            }
        }

        // Step 4: Apply weights
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.noUnusedKeys])

        // Step 5: Evaluate to materialize
        eval(model)

        print("[Gemma3] Weights loaded and evaluated successfully")
    }

    /// Load config and weights together (convenience).
    ///
    /// - Parameter directory: Model directory with config.json + safetensors
    /// - Returns: Loaded Gemma3TextModel
    static func loadModel(from directory: URL) throws -> Gemma3TextModel {
        let config = try Gemma3Config.load(from: directory)
        print("[Gemma3] Config: \(config.hiddenLayers) layers, \(config.hiddenSize) hidden, \(config.attentionHeads) heads")
        if let q = config.quantization {
            print("[Gemma3] Quantization: \(q.bits)-bit, group_size=\(q.groupSize)")
        }

        let model = Gemma3TextModel(config)
        try loadWeights(into: model, from: directory)
        return model
    }
}
