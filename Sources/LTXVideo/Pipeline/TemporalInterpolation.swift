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
        carryForward: Bool = false,
        anchorsPath: String? = nil,
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

        // Generated keyframe slots from the run that produced this clip. They
        // are anchors the model made itself, at full quality, in the same pass —
        // stronger than anchoring on a source frame, which has been through a
        // VAE round trip and a temporal upsample since.
        var slotAnchors: (latents: MLXArray, pixelFrames: [Int])? = nil
        if let anchorsPath {
            let (arrays, _) = try MLX.loadArraysAndMetadata(
                url: URL(fileURLWithPath: anchorsPath))
            guard let latents = arrays["keyframes"],
                  let indices = arrays["pixel_frame_indices"] else {
                throw LTXError.invalidConfiguration(
                    "\(anchorsPath) carries no `keyframes` / `pixel_frame_indices` — "
                    + "it is not a --slots-out file")
            }
            let frames = indices.asArray(Int32.self).map(Int.init)
            guard latents.dim(3) == height / 32, latents.dim(4) == width / 32 else {
                throw LTXError.invalidConfiguration(
                    "anchors are \(latents.dim(3))×\(latents.dim(4)) latent cells, this clip is "
                    + "\(height / 32)×\(width / 32) — they come from a different geometry")
            }
            guard latents.dim(2) == frames.count else {
                throw LTXError.invalidConfiguration(
                    "anchors: \(latents.dim(2)) keyframes for \(frames.count) indices")
            }
            slotAnchors = (latents.asType(.float32), frames)
            LTXDebug.log("[temporal] \(frames.count) generated-keyframe anchors at source "
                + "pixel frames \(frames)")
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
        var carried: MLXArray? = nil        // previous tile's denoised window

        for (index, tile) in tiles.enumerated() {
            var window = latent[0..., 0..., tile.start ..< tile.endExclusive, 0..., 0...]
            let windowShape = VideoLatentShape.fromPixelDimensions(
                batch: 1, channels: 128,
                frames: (tile.length - 1) * 8 + 1, height: height, width: width)

            var anchorContext: AppendKeyframeContext? = nil
            // Three independent sources of anchors, each with its own switch:
            // source frames (`anchorEvery`), the previous tile's output
            // (`carryForward`), and generated keyframes (`anchorsPath`). They
            // shared one gate once, so `--anchor-every 0 --anchors file` threw
            // the file away without a word.
            let wantsAnchors = effectiveAnchorEvery > 0 || carryForward || slotAnchors != nil
            if wantsAnchors, let cfg = (transformer?.config ?? ltx2Transformer?.config) {
                var guides: [AppendedGuideTokens] = []
                var anchored: [Int] = []
                var anchoredSlots: [Int] = []

                // Optional: carry the previous tile's *denoised* output into
                // this one's lead-in, anchoring on what was actually rendered
                // next door rather than on the interpolated source.
                //
                // Off by default because it lost its own bake-off. It does fix
                // the high-noise failure it was written for (identity 13.4 →
                // 23.5 dB at a seam), but the simple tiled defaults reach
                // 24.3 dB with a smaller seam spike and half the wall time —
                // our lead-in is two latent frames, too little carried signal
                // to pay for doubling the anchor count. It would come into its
                // own with upstream's multi-round structure, where far more is
                // carried; kept for that, not for today.
                if carryForward, let previous = carried, tile.dropPrefix > 0 {
                    for local in 0 ..< tile.dropPrefix {
                        let inPrevious = previous.dim(2) - tile.dropPrefix + local
                        guard inPrevious >= 0 else { continue }
                        guides.append(buildKeyframeGuideToken(
                            encodedLatent: previous[
                                0..., 0..., inPrevious ..< (inPrevious + 1), 0..., 0...],
                            temporalPosition: Self.gridTemporalPosition(
                                latentFrame: local, fps: denseFPS)))
                        anchored.append(local)
                    }
                }
                // Generated keyframes, when the caller supplied them. A slot
                // made at source pixel frame p sits at 2p once the clip is
                // densified — the frame count doubles at constant duration, the
                // way upstream scales its carried seam positions. Unlike a
                // source-frame anchor, a slot spans a single pixel frame, so it
                // is placed by pixel index rather than on the latent grid.
                if let anchors = slotAnchors {
                    for (slot, sourceFrame) in anchors.pixelFrames.enumerated() {
                        guard let localPixel = Self.slotLocalPixel(
                            sourceFrame: sourceFrame, tile: tile) else { continue }
                        guides.append(buildKeyframeGuideToken(
                            encodedLatent: anchors.latents[
                                0..., 0..., slot ..< (slot + 1), 0..., 0...],
                            pixelFrameIndex: localPixel,
                            fps: denseFPS))
                        anchoredSlots.append(localPixel)
                    }
                }

                // Source frames sit at even indices of the densified latent; a
                // tile anchors those falling inside its own window, addressed
                // locally so its RoPE grid matches.
                var positions = Set<Int>()
                if effectiveAnchorEvery > 0 {
                    positions = Set(
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
                }
                for global in positions.sorted() {
                    let local = global - tile.start
                    if anchored.contains(local) { continue }   // carried already covers it
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
                        + "anchors at local \(anchored)"
                        + (anchoredSlots.isEmpty ? ""
                           : ", generated keyframes at local pixel \(anchoredSlots)"))
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
            carried = window
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

    /// Where a generated keyframe lands inside one tile, or `nil` when it falls
    /// outside it.
    ///
    /// Two rebasings compose here, and getting either wrong puts a perfectly
    /// good anchor on the wrong moment — which reads as the model changing its
    /// mind rather than as a bug:
    ///
    /// * a round doubles the frame count at constant duration, so a slot made at
    ///   source pixel frame `p` is at `2p` in the densified clip (upstream
    ///   scales its carried seam positions the same way);
    /// * a tile is denoised standalone with positions restarting at 0, and its
    ///   first latent frame covers pixels from `8·start − 7` (the causal grid
    ///   clamps that to 0 for the first frame).
    static func slotLocalPixel(sourceFrame: Int, tile: TemporalTile) -> Int? {
        let tilePixelStart = max(8 * tile.start - 7, 0)
        let tilePixelCount = (tile.length - 1) * 8 + 1
        let local = 2 * sourceFrame - tilePixelStart
        return (local >= 0 && local < tilePixelCount) ? local : nil
    }

    /// Temporal coordinate the position grid gives a latent frame — the
    /// midpoint of the pixel span it covers, after the causal shift, over fps.
    /// Anchors have to land on exactly this, not on a rounded pixel index.
    static func gridTemporalPosition(latentFrame i: Int, fps: Float = 24.0, temporalScale: Float = 8) -> Float {
        let start = max(Float(i) * temporalScale + (1 - temporalScale), 0)
        let end = max((Float(i) + 1) * temporalScale + (1 - temporalScale), 0)
        return ((start + end) / 2) / fps
    }


}
