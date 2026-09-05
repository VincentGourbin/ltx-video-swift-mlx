// LTXCheckpointSource.swift - Unified and split checkpoint layouts
// Copyright 2025

import Foundation
@preconcurrency import MLX

/// Resolved on-disk location of every weight file a model variant needs.
///
/// LTX-2.3 packs transformer, connectors and video VAE into one file, so all
/// three fields point at it. LTX-2.5 ships one file per component and adds the
/// text encoder, which for 2.3 is an external Gemma 3 directory instead.
public struct LTXCheckpointPaths: Sendable, Equatable {
    /// File holding the diffusion transformer and both embedding connectors.
    public let transformer: URL

    /// File holding the video VAE (encoder + decoder + per-channel statistics).
    public let videoVAE: URL

    /// LTX-2.5 text-encoder bundle (Gemma 4 + tokenizer + LTX projections).
    /// `nil` for LTX-2.3, whose encoder is an external Gemma 3 checkpoint.
    public let textEncoder: URL?

    /// File carrying `vocoder.*` — the BigVGAN generator plus its bandwidth
    /// extension. Unified checkpoints keep it alongside everything else; split
    /// ones put it in the audio-VAE component.
    public let audioBundle: URL

    public init(transformer: URL, videoVAE: URL, textEncoder: URL? = nil, audioBundle: URL? = nil) {
        self.transformer = transformer
        self.videoVAE = videoVAE
        self.textEncoder = textEncoder
        self.audioBundle = audioBundle ?? transformer
    }
}

/// Loads a model variant's weights, whatever layout its generation uses.
///
/// Both layouts are normalised into the three buckets the pipeline applies —
/// transformer, video VAE decoder, text-side connector — so nothing downstream
/// has to know which generation it is running.
///
/// The one structural difference worth stating: in LTX-2.3 the aggregate
/// projections (`text_embedding_projection.*`) sit in the unified file, while in
/// LTX-2.5 they ship inside the text-encoder bundle. The connector blocks stay
/// with the transformer in both.
struct LTXCheckpointSource {
    let model: LTXModel
    let paths: LTXCheckpointPaths

    /// Weights for the transformer, the video VAE decoder and the text-side stack.
    ///
    /// - Parameter precomputedProjectionWeights: the LTX aggregate projections,
    ///   already extracted from a text-encoder bundle. Prefer this over
    ///   `textEncoderAssets` when the caller already parsed that bundle for
    ///   something else (typically the Gemma load): `textEncoderAssets` wraps
    ///   the whole multi-GB file, and this function reading it a second time
    ///   here would keep that whole file alive for as long as the caller's
    ///   reference to it lives — see
    ///   `docs/knowledge/pitfalls/quantized-load-materialised-full-bf16.md`.
    func loadComponents(
        includeAudio: Bool = false,
        textEncoderAssets: LTX25TextEncoderAssets? = nil,
        precomputedProjectionWeights: [String: MLXArray]? = nil
    ) throws -> (transformer: [String: MLXArray], vae: [String: MLXArray], connector: [String: MLXArray]) {
        switch model.weightsLayout {
        case .unified:
            return try LTXWeightLoader.splitUnifiedWeightsFile(
                path: paths.transformer.path, includeAudio: includeAudio)

        case .split:
            // The transformer shard carries no VAE tensors, so that bucket comes
            // back empty and is filled from the VAE file.
            let (transformer, _, connector) = try LTXWeightLoader.splitUnifiedWeightsFile(
                path: paths.transformer.path, includeAudio: includeAudio)
            let vae = try LTXWeightLoader.loadVAEWeights(from: paths.videoVAE.path)

            var textSide = connector
            textSide.merge(try loadProjectionWeights(
                textEncoderAssets: textEncoderAssets,
                precomputed: precomputedProjectionWeights)
            ) { current, _ in current }
            return (transformer, vae, textSide)
        }
    }

    /// VAE encoder weights (image conditioning, retake, IC-LoRA references).
    func loadVAEEncoderWeights() throws -> [String: MLXArray] {
        switch model.weightsLayout {
        case .unified:
            return try LTXWeightLoader.loadVAEEncoderWeightsFromUnified(from: paths.transformer.path)
        case .split:
            return try LTXWeightLoader.loadVAEEncoderWeights(from: paths.videoVAE.path)
        }
    }

    /// LTX aggregate projections. Split layouts keep them with the text encoder.
    private func loadProjectionWeights(
        textEncoderAssets: LTX25TextEncoderAssets? = nil,
        precomputed: [String: MLXArray]? = nil
    ) throws -> [String: MLXArray] {
        if let precomputed { return precomputed }
        if let textEncoderAssets { return textEncoderAssets.projectionWeights() }
        guard let textEncoder = paths.textEncoder else {
            throw LTXError.weightLoadingFailed(
                "\(model.displayName) is a split checkpoint but no text-encoder path was resolved")
        }
        return try LTX25TextEncoderAssets(fileURL: textEncoder).projectionWeights()
    }

    /// The transformer file's safetensors metadata, used to check that the text
    /// encoder is the one this checkpoint was trained against.
    func transformerMetadata() throws -> [String: String] {
        try LoRALoader.loadMetadata(from: paths.transformer.path)
    }
}
