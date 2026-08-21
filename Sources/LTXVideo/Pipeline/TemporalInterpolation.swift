// TemporalInterpolation.swift - Frame-rate doubling through the temporal upsampler
// Copyright 2026

import Foundation
@preconcurrency import MLX

extension LTXPipeline {

    /// Double a clip's frame rate: `n` frames become `2n - 1`, at the same
    /// resolution and duration.
    ///
    /// Two steps, and the second is what makes it worth doing. The latent
    /// temporal upsampler lays the existing motion onto a denser frame grid —
    /// on its own that is an interpolation, and it looks like one. A short
    /// refinement pass then re-denoises the whole clip with an **ancestral**
    /// sampler, whose re-injected noise lets the model invent plausible
    /// in-between motion rather than average the frames it already has.
    ///
    /// ## Scope
    ///
    /// Long canvases are denoised in overlapping tiles, each anchored on the
    /// source frames inside its own window and contributing only the frames it
    /// owns — so memory follows the tile budget, not the clip length. What is
    /// still missing relative to upstream's `DFRPipeline` is the *generated*
    /// mid-segment keyframe slots and the spatial detailing LoRA.
    ///
    /// - Parameters:
    ///   - videoPath: the clip to densify. Its frame count must be `8n + 1`.
    ///   - prompt: still conditions the refinement — pass what the clip depicts.
    ///   - upscalerPath: the *temporal* upscaler; a spatial one is refused.
    ///   - strength: how far to renoise before refining. Higher invents more
    ///     motion and drifts further from the source; the default matches
    ///     upstream's temporal rounds.
    public func interpolateTemporally(
        videoPath: String,
        prompt: String,
        upscalerPath: String,
        width: Int,
        height: Int,
        numFrames: Int,
        seed: UInt64? = nil,
        eta: Float = 0.5,
        renoiseFrom: Float? = nil,
        anchorEvery: Int? = nil,
        maxTileLatentFrames: Int = 32,
        sourceFPS: Float = 24.0,
        onProgress: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> VideoGenerationResult {
        // A round doubles the frame count at constant duration, so the refined
        // clip runs at twice the source rate — and must be positioned at that
        // rate. Upstream caps conditioning at 60 fps; past two rounds from 24
        // the cap is what the model actually sees.
        let denseFPS = min(sourceFPS * 2, 60.0)
        let startTime = Date()
        guard FileManager.default.fileExists(atPath: videoPath) else {
            throw LTXError.fileNotFound("Video not found: \(videoPath)")
        }
        guard (numFrames - 1) % 8 == 0 else {
            throw LTXError.invalidConfiguration("Frame count must be 8n+1; got \(numFrames)")
        }
        if !isLoaded { try await loadModels(progressCallback: nil) }
        let beacon = RuntimeBeacon.begin(task: "temporal-interpolate", model: model.rawValue)
        defer { beacon?.end() }

        let upscaler = try loadTemporalUpscaler(from: upscalerPath)

        // Encode the source clip, then densify its latent.
        let sourceLatent = try await encodeVideo(
            path: videoPath, width: width, height: height, numFrames: numFrames)
        onProgress?(GenerationProgress(
            currentStep: 0, totalSteps: temporalSigmas.count, sigma: 1, phase: .upscaling))

        let decoderStats = vaeDecoder
        let mean = (decoderStats?.meanOfMeans ?? MLXArray.zeros([128])).reshaped([1, -1, 1, 1, 1])
        let std = (decoderStats?.stdOfMeans ?? MLXArray.ones([128])).reshaped([1, -1, 1, 1, 1])
        // The upsampler works on un-normalised latents, like the spatial one.
        var latent = (upscaler(sourceLatent * std + mean) - mean) / std
        MLX.eval(latent)
        let densifiedFrames = (latent.dim(2) - 1) * 8 + 1
        LTXDebug.log("[temporal] \(numFrames) → \(densifiedFrames) frames, latent \(latent.shape)")

        // Refine: renoise to the first sigma, then walk the schedule ancestrally.
        let encoded = try await encodeText(prompt)
        unloadGemmaIfConfigured()

        // The anchors must show the *clean* densified latent, not the renoised one.
        let sourceLatentDensified = latent
        if let seed { MLXRandom.seed(seed) }

        let tiles = TemporalTiling.tiles(
            latentFrames: latent.dim(2), maxTileFrames: maxTileLatentFrames)

        // Tiling changes what is safe. A single window can renoise to 0.975 and
        // anchor sparsely: the anchors only have to hold one continuous
        // trajectory. Tiles renoise *independently*, so at that level each one
        // rebuilds its own subject from near-noise and the seams stop agreeing —
        // measured 13.4 dB identity at a seam, against 24.3 dB with the tiled
        // defaults below. Callers can still override both.
        let tiled = tiles.count > 1
        let effectiveRenoise = renoiseFrom ?? (tiled ? 0.725 : 0.975)
        let effectiveAnchorEvery = anchorEvery ?? (tiled ? 1 : 4)
        if tiled {
            LTXDebug.log("[temporal] \(tiles.count) tiles over \(latent.dim(2)) latent frames — "
                + "renoise \(effectiveRenoise), anchor every \(effectiveAnchorEvery)"
                + (renoiseFrom != nil || anchorEvery != nil ? " (caller override)" : " (tiled defaults)"))
        }

        let sigmas = temporalSigmas.filter { $0 <= effectiveRenoise || $0 == 0 }
        guard sigmas.count >= 2 else {
            throw LTXError.invalidConfiguration(
                "renoiseFrom \(effectiveRenoise) leaves no refinement steps; use at least 0.42")
        }
        LTXDebug.log("[temporal] refining from σ=\(sigmas[0]) over \(sigmas.count - 1) steps")

        // Renoise the whole canvas once, at the level the schedule starts from.
        latent = MLXArray(sigmas[0]) * MLXRandom.normal(latent.shape).asType(latent.dtype)
            + MLXArray(1.0 - sigmas[0]) * latent
        MLX.eval(latent)

        // Refine tile by tile. Each tile is denoised as a standalone sequence
        // (positions restart at 0, as upstream's remap does), anchored on the
        // source frames inside its own window, then contributes only the frames
        // it owns — its lead-in was denoised solely to carry motion across the
        // seam it shares with the previous tile.
        let stepper = AncestralEulerStep(eta: eta)
        var refined: [MLXArray] = []

        for (index, tile) in tiles.enumerated() {
            var window = latent[0..., 0..., tile.start ..< tile.endExclusive, 0..., 0...]
            let windowShape = VideoLatentShape.fromPixelDimensions(
                batch: 1, channels: 128,
                frames: (tile.length - 1) * 8 + 1, height: height, width: width)

            var anchorContext: AppendKeyframeContext? = nil
            if effectiveAnchorEvery > 0, let cfg = (transformer?.config ?? ltx2Transformer?.config) {
                var guides: [AppendedGuideTokens] = []
                var anchored: [Int] = []
                // Source frames sit at even indices of the densified latent; a
                // tile anchors those falling inside its own window, addressed
                // locally so its RoPE grid matches.
                var positions = Set(
                    stride(from: 0, to: latent.dim(2), by: 2 * effectiveAnchorEvery)
                        .filter { $0 >= tile.start && $0 < tile.endExclusive })
                // Always anchor the seam itself. A strided anchor grid does not
                // generally land on a tile boundary, and a weakly anchored seam
                // is exactly where the two tiles' inventions fail to meet:
                // measured a 38.4 inter-frame spike (z = +11) at a seam with no
                // anchor within four frames, against none at a seam that had one.
                // Source frames sit at even indices, so round the boundary down.
                for boundary in [tile.start, tile.start + tile.dropPrefix]
                where boundary < tile.endExclusive {
                    positions.insert(boundary - (boundary % 2))
                }
                for global in positions.sorted() {
                    let local = global - tile.start
                    guides.append(buildKeyframeGuideToken(
                        encodedLatent: sourceLatentDensified[
                            0..., 0..., global ..< (global + 1), 0..., 0...],
                        temporalPosition: Self.gridTemporalPosition(
                            latentFrame: local, fps: denseFPS)))
                    anchored.append(local)
                }
                if !guides.isEmpty {
                    anchorContext = assembleAppendContext(
                        guides: guides, shape: windowShape, hasAudio: false,
                        refConfig: cfg, stageLabel: "tile \(index) anchors", fps: denseFPS)
                    LTXDebug.log("[temporal] tile \(index) frames \(tile.start)..<\(tile.endExclusive), "
                        + "anchors at local \(anchored)")
                }
            }

            for step in 0 ..< (sigmas.count - 1) {
                let sigma = sigmas[step]
                onProgress?(GenerationProgress(
                    currentStep: index * (sigmas.count - 1) + step,
                    totalSteps: tiles.count * (sigmas.count - 1),
                    sigma: sigma, phase: .refinement))
                let velocity = runDenoiseStep(
                    sigma: sigma, videoLatent: window, audioLatentPacked: nil,
                    shape: windowShape, videoAppendCtx: anchorContext,
                    audioRefCtx: nil, audioNumFrames: 0,
                    videoTextEmbeddings: encoded.embeddings,
                    audioTextEmbeddings: encoded.embeddings,
                    textMask: encoded.mask)
                // The transformer predicts velocity; the ancestral step wants x₀.
                let denoised = window - MLXArray(sigma) * velocity.video
                window = stepper(
                    sample: window, denoised: denoised,
                    sigma: sigma, sigmaNext: sigmas[step + 1],
                    noise: MLXRandom.normal(window.shape).asType(window.dtype))
                MLX.eval(window)
            }

            refined.append(window[0..., 0..., tile.dropPrefix ..< tile.length, 0..., 0...])
        }

        latent = refined.count == 1 ? refined[0] : MLX.concatenated(refined, axis: 2)
        MLX.eval(latent)

        onProgress?(GenerationProgress(
            currentStep: sigmas.count - 1, totalSteps: sigmas.count - 1, sigma: 0, phase: .decoding))
        let frames = decodeFrames(latent: latent)
        MLX.eval(frames)

        return VideoGenerationResult(
            frames: frames,
            seed: seed ?? 0,
            generationTime: Date().timeIntervalSince(startTime),
            audioWaveform: nil, audioSampleRate: nil,
            effectivePrompt: prompt)
    }

    /// The tail of the distilled schedule, which upstream's temporal rounds use:
    /// four steps starting below the high-noise region, since the input already
    /// carries the composition.
    var temporalSigmas: [Float] { Array(DISTILLED_SIGMA_VALUES.dropFirst(4)) }

    /// Temporal coordinate the position grid gives a latent frame — the
    /// midpoint of the pixel span it covers, after the causal shift, over fps.
    /// Anchors have to land on exactly this, not on a rounded pixel index.
    static func gridTemporalPosition(latentFrame i: Int, fps: Float = 24.0, temporalScale: Float = 8) -> Float {
        let start = max(Float(i) * temporalScale + (1 - temporalScale), 0)
        let end = max((Float(i) + 1) * temporalScale + (1 - temporalScale), 0)
        return ((start + end) / 2) / fps
    }


}
