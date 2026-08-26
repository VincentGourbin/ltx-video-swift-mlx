// LTXDurationHead.swift - Caption-driven clip length (LTX-2.5)
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Predicts a clip's natural duration from the connector's caption tokens.
///
/// New in LTX-2.5 and optional: when the caller omits a frame count, this head
/// picks one from the prompt instead of a fixed default. It is a 3.8 MB
/// regression head over the *already computed* connector output, so asking it
/// costs one small forward — no diffusion, no second text encode.
///
/// Trained in log-seconds so the loss spreads evenly across orders of magnitude;
/// ``predictSeconds`` exponentiates, and ``predictFrameCount`` snaps the result
/// to the VAE's `8k + 1` temporal grid.
final class LTXDurationHead: Module {
    @ModuleInfo(key: "video_input_proj") var videoInputProj: Linear
    @ModuleInfo(key: "audio_input_proj") var audioInputProj: Linear
    @ModuleInfo(key: "attention_pooler") var attentionPooler: AttentionPooler
    @ModuleInfo(key: "mlp_hidden") var mlpHidden: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    @ParameterInfo(key: "video_modality_emb") var videoModalityEmbedding: MLXArray
    @ParameterInfo(key: "audio_modality_emb") var audioModalityEmbedding: MLXArray

    init(
        videoCrossAttentionDim: Int = 4096,
        audioCrossAttentionDim: Int = 2048,
        poolerHiddenDim: Int = 256,
        numQueries: Int = 1,
        numPoolerHeads: Int = 4,
        mlpHiddenDim: Int = 256
    ) {
        self._videoInputProj.wrappedValue = Linear(videoCrossAttentionDim, poolerHiddenDim)
        self._audioInputProj.wrappedValue = Linear(audioCrossAttentionDim, poolerHiddenDim)
        self._attentionPooler.wrappedValue = AttentionPooler(
            hiddenDim: poolerHiddenDim, numQueries: numQueries, numHeads: numPoolerHeads)
        self._mlpHidden.wrappedValue = Linear(poolerHiddenDim * numQueries, mlpHiddenDim)
        self._mlpOut.wrappedValue = Linear(mlpHiddenDim, 1)
        self._videoModalityEmbedding.wrappedValue = MLXArray.zeros([poolerHiddenDim])
        self._audioModalityEmbedding.wrappedValue = MLXArray.zeros([poolerHiddenDim])
        super.init()
    }

    /// Predicted duration in seconds for a single-item batch.
    ///
    /// No attention mask: the connector substitutes learnable registers for padded
    /// positions and marks the result fully attendable, so every token here is valid.
    func predictSeconds(videoTokens: MLXArray?, audioTokens: MLXArray?) throws -> Float {
        var groups: [MLXArray] = []
        if let videoTokens {
            groups.append(videoInputProj(videoTokens) + videoModalityEmbedding)
        }
        if let audioTokens {
            groups.append(audioInputProj(audioTokens) + audioModalityEmbedding)
        }
        guard !groups.isEmpty else {
            throw LTXError.invalidConfiguration(
                "The duration head needs video or audio connector tokens")
        }

        let tokens = groups.count == 1 ? groups[0] : MLX.concatenated(groups, axis: 1)
        let pooled = attentionPooler(tokens)
        let flattened = pooled.reshaped([pooled.dim(0), -1])
        let hidden = MLXNN.geluApproximate(mlpHidden(flattened))
        let logDuration = mlpOut(hidden).squeezed(axis: -1)
        let seconds = MLX.exp(logDuration)
        MLX.eval(seconds)
        return seconds.item(Float.self)
    }

    /// Predicted frame count, clamped and snapped to the `8k + 1` grid.
    ///
    /// The clamp is a safety rail, not a preference: an outlier prediction would
    /// otherwise request a degenerate or OOM-sized generation.
    func predictFrameCount(
        videoTokens: MLXArray?,
        audioTokens: MLXArray?,
        frameRate: Float = 24.0,
        minSeconds: Float = 1.0,
        maxSeconds: Float = 20.0
    ) throws -> (frames: Int, rawSeconds: Float, wasClamped: Bool) {
        let seconds = try predictSeconds(videoTokens: videoTokens, audioTokens: audioTokens)
        let frames = Self.snapToGrid(
            seconds: seconds, frameRate: frameRate,
            minSeconds: minSeconds, maxSeconds: maxSeconds)
        let clamped = seconds > maxSeconds || seconds < minSeconds
        return (frames, seconds, clamped)
    }

    /// Pure grid arithmetic behind ``predictFrameCount`` — always returns 8k+1.
    static func snapToGrid(
        seconds: Float, frameRate: Float, minSeconds: Float, maxSeconds: Float
    ) -> Int {
        let minFrames = Int((minSeconds * frameRate).rounded())
        let maxFrames = Int((maxSeconds * frameRate).rounded())

        // `.toNearestOrEven`, not `.rounded()`: upstream converts with Python's
        // round(), which is banker's rounding. At 5.6875 s the product is exactly
        // 136.5 — half-away-from-zero gives 137 frames, banker's gives 129, a
        // full grid step apart. Pinned by the 5.6875 row of the parity table.
        var raw = Int(((seconds.isFinite ? seconds : 0) * frameRate).rounded(.toNearestOrEven))
        raw = max(minFrames, min(raw, maxFrames))

        // Rounds **down**, unlike FrameCountSpec's `.nearest`, and deliberately:
        // this is a prediction to stay within, not a duration someone asked for.
        // The gap is a documented contract — see GridRounding.
        let timeScale = FrameGrid.step
        var frames = FrameGrid.snap(raw, rounding: .down)
        if frames < minFrames {
            let rounded = ((minFrames - 1) + timeScale - 1) / timeScale * timeScale + 1
            // A [min, max] window may contain no 8k+1 point at all; capping the
            // snap-up at maxFrames would leave the grid. Prefer the grid — the
            // whole contract is "safe to hand to LTXVideoGenerationConfig" —
            // and exceed maxFrames by at most timeScale-1 frames in that case.
            frames = rounded <= maxFrames ? rounded : frames + timeScale
        }
        return frames
    }

    // MARK: - Loading

    /// Load from `model_patches/ltx-2.5-duration-head-bf16.safetensors`.
    static func load(from url: URL) throws -> LTXDurationHead {
        let raw = try MLX.loadArrays(url: url)
        let prefix = "duration_head."

        var weights: [String: MLXArray] = [:]
        for (key, value) in raw where key.hasPrefix(prefix) {
            weights[String(key.dropFirst(prefix.count))] = value
        }
        guard !weights.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "No duration-head weights in \(url.lastPathComponent)")
        }

        let head = LTXDurationHead()
        let declared = Set(head.parameters().flattened().map(\.0))
        let missing = declared.subtracting(weights.keys)
        guard missing.isEmpty else {
            throw LTXError.weightLoadingFailed(
                "Duration head: \(missing.count) parameters unfed (\(missing.sorted().joined(separator: ", ")))")
        }

        head.update(parameters: ModuleParameters.unflattened(weights))
        eval(head.parameters())
        return head
    }
}

// MARK: - Attention Pooler

/// Cross-attends `numQueries` learnable tokens against the connector output,
/// producing a fixed-size summary whatever the sequence length.
final class AttentionPooler: Module {
    @ParameterInfo(key: "query_tokens") var queryTokens: MLXArray
    @ModuleInfo(key: "cross_attn") var crossAttention: PackedMultiheadAttention

    let hiddenDim: Int
    let numQueries: Int

    init(hiddenDim: Int = 256, numQueries: Int = 1, numHeads: Int = 4) {
        self.hiddenDim = hiddenDim
        self.numQueries = numQueries
        self._queryTokens.wrappedValue = MLXArray.zeros([numQueries, hiddenDim])
        self._crossAttention.wrappedValue = PackedMultiheadAttention(
            embedDim: hiddenDim, numHeads: numHeads)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let batch = tokens.dim(0)
        let queries = MLX.broadcast(
            queryTokens.expandedDimensions(axis: 0), to: [batch, numQueries, hiddenDim])
        return crossAttention(queries: queries, keys: tokens, values: tokens)
    }
}

/// `torch.nn.MultiheadAttention` with `batch_first=True`, weights kept in the
/// checkpoint's packed layout: one `in_proj_weight` of `[3 * embedDim, embedDim]`
/// covering q, k and v, and a separate `out_proj`.
final class PackedMultiheadAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    private let embedDim: Int

    init(embedDim: Int, numHeads: Int) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self._inProjWeight.wrappedValue = MLXArray.zeros([3 * embedDim, embedDim])
        self._inProjBias.wrappedValue = MLXArray.zeros([3 * embedDim])
        self._outProj.wrappedValue = Linear(embedDim, embedDim)
        super.init()
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let qWeight = inProjWeight[0 ..< embedDim]
        let kWeight = inProjWeight[embedDim ..< (2 * embedDim)]
        let vWeight = inProjWeight[(2 * embedDim)...]
        let qBias = inProjBias[0 ..< embedDim]
        let kBias = inProjBias[embedDim ..< (2 * embedDim)]
        let vBias = inProjBias[(2 * embedDim)...]

        let q = MLX.matmul(queries, qWeight.transposed()) + qBias
        let k = MLX.matmul(keys, kWeight.transposed()) + kBias
        let v = MLX.matmul(values, vWeight.transposed()) + vBias

        func split(_ x: MLXArray) -> MLXArray {
            x.reshaped([x.dim(0), x.dim(1), numHeads, headDim]).transposed(0, 2, 1, 3)
        }

        let scale = 1.0 / sqrt(Float(headDim))
        var attended = MLXFast.scaledDotProductAttention(
            queries: split(q), keys: split(k), values: split(v), scale: scale, mask: nil)
        attended = attended.transposed(0, 2, 1, 3)
            .reshaped([queries.dim(0), queries.dim(1), embedDim])
        return outProj(attended)
    }
}
