// Gemma4TextEncoder.swift - LTX-2.5 prompt encoding on Gemma 4
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN
import Gemma4Swift
import Tokenizers

/// Produces the 49 Gemma hidden states that LTX's feature extractor consumes.
///
/// Implemented twice: `Gemma3TextModel` for LTX-2.3 (stock Gemma 3, ported in
/// this package) and ``Gemma4TextEncoder`` for LTX-2.5 (`gemma4-12b-ltx-v1`,
/// bundled with the checkpoint and run through `Gemma4Swift`). The pipeline
/// talks to this protocol so the two generations share every downstream stage.
protocol LTXGemmaEncoding: AnyObject {
    /// Number of transformer layers; the hidden-state count is this + 1.
    var numHiddenLayers: Int { get }

    /// Encode a prompt into `numHiddenLayers + 1` hidden states plus the
    /// attention mask that marks real tokens.
    ///
    /// - Parameters:
    ///   - prompt: raw prompt text.
    ///   - maxLength: sequence length the states are padded to (LTX uses 1024).
    /// - Returns: `states` of `[1, maxLength, hidden]` and a `[1, maxLength]` mask.
    func encode(prompt: String, maxLength: Int) throws -> (states: [MLXArray], attentionMask: MLXArray)
}

/// LTX-2.5 text encoder: the bundled Gemma 4 run as a pure encoder.
///
/// ## Padding
///
/// `Gemma4Swift`'s forward takes no per-sequence attention mask, so this encoder
/// runs the prompt **unpadded** at batch 1 and left-pads the resulting hidden
/// states with zeros. That is not a shortcut, it is exact: attention is causal,
/// so a real token never attends to a left-pad slot that a mask would have
/// blocked anyway, and Gemma's RoPE is relative — shifting every position by the
/// same amount leaves the attention scores unchanged. The padded slots are then
/// zeroed by the feature extractor's per-token RMS path regardless of what they
/// hold. Feeding padded ids instead would let real tokens attend to pad slots.
final class Gemma4TextEncoder: LTXGemmaEncoding {
    private let model: Gemma4LLMModel
    private let tokenizer: any Tokenizers.Tokenizer
    private let bosTokenID: Int32

    let numHiddenLayers: Int

    /// The LTX aggregate projections shipped in the same file, ready to be applied
    /// to a ``GemmaFeaturesExtractor``.
    let projectionWeights: [String: MLXArray]

    private init(
        model: Gemma4LLMModel,
        tokenizer: any Tokenizers.Tokenizer,
        bosTokenID: Int32,
        numHiddenLayers: Int,
        projectionWeights: [String: MLXArray]
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.bosTokenID = bosTokenID
        self.numHiddenLayers = numHiddenLayers
        self.projectionWeights = projectionWeights
    }

    // MARK: - Loading

    /// Load the encoder from an LTX-2.5 text-encoder safetensors file.
    ///
    /// - Parameters:
    ///   - fileURL: `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`.
    ///   - tokenizerCacheDirectory: small directory the bundled tokenizer assets
    ///     are extracted into (they live inside the safetensors as byte tensors).
    ///   - transformerMetadata: when provided, the encoder is checked against the
    ///     transformer's `gemma_source_checkpoint`.
    ///   - quantization: optional on-the-fly quantization; the checkpoint is bf16
    ///     only (~24 GB resident), and no community quantisation of this
    ///     derivative exists to load instead.
    static func load(
        fileURL: URL,
        tokenizerCacheDirectory: URL,
        transformerMetadata: [String: String]? = nil,
        quantization: TransformerQuantization? = nil
    ) async throws -> Gemma4TextEncoder {
        try await load(
            assets: LTX25TextEncoderAssets(fileURL: fileURL),
            tokenizerCacheDirectory: tokenizerCacheDirectory,
            transformerMetadata: transformerMetadata,
            quantization: quantization)
    }

    /// Same, from an already-parsed bundle — the pipeline parses the ~24 GB
    /// file once and shares it between the encoder and the projections.
    static func load(
        assets: LTX25TextEncoderAssets,
        tokenizerCacheDirectory: URL,
        transformerMetadata: [String: String]? = nil,
        quantization: TransformerQuantization? = nil
    ) async throws -> Gemma4TextEncoder {
        if let transformerMetadata {
            try assets.verifyPairing(withTransformerMetadata: transformerMetadata)
        }

        let config = try assets.textConfig()
        LTXDebug.log(
            "[Gemma4] \(assets.gemmaVersion ?? "unknown"): \(config.numHiddenLayers) layers, "
            + "hidden \(config.hiddenSize)")

        // Pull everything else out of `assets` *before* touching the Gemma
        // weights below, so nothing past this point still needs the whole
        // ~26 GB bundle a second time (see loadModels()/loadTextEncoderModels()
        // in LTXPipeline.swift for the matching caller-side change).
        let tokenizerDirectory = try assets.materializeTokenizer(in: tokenizerCacheDirectory)
        let projectionWeights = assets.projectionWeights()

        let model = Gemma4LLMModel(config: config)
        var gemmaWeights = assets.gemmaWeights()
        try applyWeights(gemmaWeights, to: model)
        gemmaWeights.removeAll()

        if let quantization, quantization != .bf16 {
            try quantize(model, with: quantization)
        }
        // NOTE (2026-09-05, investigating issue #86): on-the-fly quantization
        // of this checkpoint does not currently reduce its resident memory —
        // measured additive, not replacing: bf16 alone evaluates to ~22.7 GB,
        // int4 evaluates to ~29.1 GB (bf16 + ~6.2 GB of quantized weights on
        // top), regardless of eval ordering or granularity (whole-model,
        // count-chunked, or per-layer via `loraLayers` all measured
        // identical). The likely cause is upstream of this repo — see
        // docs/knowledge/pitfalls/gemma4-quantize-does-not-release-bf16.md.
        // A single eval is kept here (no memory benefit was found from
        // chunking) pending a fix.
        MLX.eval(model.parameters())

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)

        // Gemma 4's tokenizer does not emit BOS from its post-processor (Gemma 3 did),
        // so the encode path prepends it explicitly — as the reference implementation does.
        let bos = tokenizer.bosToken.flatMap { tokenizer.convertTokenToId($0) } ?? 2

        return Gemma4TextEncoder(
            model: model,
            tokenizer: tokenizer,
            bosTokenID: Int32(bos),
            numHiddenLayers: config.numHiddenLayers,
            projectionWeights: projectionWeights
        )
    }

    /// Apply the bundle's weights, reporting anything that did not land.
    ///
    /// `Gemma4Swift`'s sanitizer handles the checkpoint-shape details (KV-shared
    /// layers, MoE splits, conv transposes); this only filters to parameters the
    /// built model actually declares, so the missing/unmatched counts are
    /// meaningful rather than drowned in the bundle's non-Gemma tensors.
    private static func applyWeights(_ weights: [String: MLXArray], to model: Gemma4LLMModel) throws {
        let sanitized = model.sanitize(weights: weights)
        let declared = Set(model.parameters().flattened().map(\.0))

        var updates: [String: MLXArray] = [:]
        var unmatched: [String] = []
        for (key, value) in sanitized {
            if declared.contains(key) {
                updates[key] = value
            } else {
                unmatched.append(key)
            }
        }

        let missing = declared.subtracting(updates.keys).sorted()
        if !missing.isEmpty {
            // A Gemma parameter left at its random initialisation silently corrupts
            // every prompt embedding, so this is fatal rather than a warning.
            throw LTXError.weightLoadingFailed(
                "Gemma 4 encoder: \(missing.count) parameters absent from the checkpoint "
                + "(\(missing.prefix(5).joined(separator: ", "))\(missing.count > 5 ? ", …" : ""))")
        }
        if !unmatched.isEmpty {
            LTXDebug.log("[Gemma4] \(unmatched.count) checkpoint keys unused: "
                + unmatched.sorted().prefix(5).joined(separator: ", "))
        }

        model.update(parameters: ModuleParameters.unflattened(updates))
        LTXDebug.log("[Gemma4] applied \(updates.count) weights")
    }

    private static func quantize(_ model: Gemma4LLMModel, with quantization: TransformerQuantization) throws {
        let bits = quantization.bits
        guard bits < 16 else { return }
        MLXNN.quantize(model: model, groupSize: quantization.groupSize, bits: bits)
        LTXDebug.log("[Gemma4] quantized to \(bits)-bit, group \(quantization.groupSize)")
    }

    // MARK: - Encoding

    /// Token ids for a prompt: BOS-prefixed, truncated, never padded.
    func tokenize(_ prompt: String, maxLength: Int) -> [Int32] {
        var ids = tokenizer.encode(text: prompt.trimmingCharacters(in: .whitespacesAndNewlines))
            .map { Int32($0) }
        if ids.first != bosTokenID {
            ids.insert(bosTokenID, at: 0)
        }
        if ids.count > maxLength {
            print("⚠️ Prompt exceeds \(maxLength) tokens (\(ids.count)); truncating the tail")
            ids = Array(ids.prefix(maxLength))
        }
        return ids
    }

    func encode(prompt: String, maxLength: Int) throws -> (states: [MLXArray], attentionMask: MLXArray) {
        let ids = tokenize(prompt, maxLength: maxLength)
        guard !ids.isEmpty else {
            throw LTXError.textEncodingFailed("Prompt tokenized to nothing")
        }

        let inputs = MLXArray(ids).reshaped([1, ids.count])
        let states = model.forwardCollectingHiddenStates(inputs)
        guard states.count == numHiddenLayers + 1 else {
            throw LTXError.textEncodingFailed(
                "Expected \(numHiddenLayers + 1) hidden states, got \(states.count)")
        }
        MLX.eval(states)

        let padding = maxLength - ids.count
        let padded = padding > 0 ? states.map { Self.leftPad($0, by: padding) } : states
        MLX.eval(padded)

        let mask = [Float](repeating: 0, count: padding) + [Float](repeating: 1, count: ids.count)
        return (padded, MLXArray(mask).reshaped([1, maxLength]))
    }

    /// Left-pad `[1, T, D]` with `count` zero rows, matching LTX's padding side.
    private static func leftPad(_ state: MLXArray, by count: Int) -> MLXArray {
        let zeros = MLXArray.zeros([state.dim(0), count, state.dim(2)], dtype: state.dtype)
        return MLX.concatenated([zeros, state], axis: 1)
    }

}
