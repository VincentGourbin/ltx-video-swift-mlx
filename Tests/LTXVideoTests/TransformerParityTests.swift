// TransformerParityTests.swift — the port's forward pass against Lightricks' own
// Copyright 2026
//
// Ground truth comes from `scripts/transformer_reference.py`, which runs
// upstream's `LTXModel` on a small config with fixed weights and inputs:
//
//   PYTHONPATH=<ltx-core>/src python3 scripts/transformer_reference.py ref.safetensors
//   TEST_RUNNER_LTX_TRANSFORMER_REFERENCE=$PWD/ref.safetensors xcodebuild ... test \
//     -only-testing:LTXVideoTests/TransformerParityTests
//
// The real checkpoint is 22B parameters, which no CPU float32 reference can
// hold. The arithmetic under test is width-independent, so a small model
// exercises every piece of it: RoPE, AdaLN-Single, qk-normed self-attention,
// cross-attention, the GELU-approximate FFN and the scale-shift output.

import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
@testable import LTXVideo

@Suite("Transformer parity",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_TRANSFORMER_REFERENCE"] != nil))
struct TransformerParityTests {

    static var referencePath: String? {
        ProcessInfo.processInfo.environment["LTX_TRANSFORMER_REFERENCE"]
    }

    /// The 9-value block every shipped 2.3/2.5 checkpoint uses. Written by
    /// `transformer_reference.py <path> ltx23`; skipped when unset.
    static var referencePathLTX23: String? {
        ProcessInfo.processInfo.environment["LTX_TRANSFORMER_REFERENCE_LTX23"]
    }

    /// Mirrors the reference script's constants.
    static func config(crossAttentionAdaLN: Bool) -> LTXTransformerConfig {
        LTXTransformerConfig(
            numLayers: 2, numAttentionHeads: 2, attentionHeadDim: 8,
            inChannels: 4, outChannels: 4,
            crossAttentionDim: 16, captionChannels: 16,
            ropeTheta: 10000.0, maxPos: [20, 2048, 2048],
            timestepScaleMultiplier: 1000, normEps: 1e-6,
            gatedAttention: crossAttentionAdaLN,
            crossAttentionAdaLN: crossAttentionAdaLN)
    }
    static let shape = (frames: 2, height: 2, width: 3)
    static let sigma: Float = 0.7

    @Test func forwardMatchesUpstream() throws {
        try compare(referencePath: Self.referencePath!, crossAttentionAdaLN: false)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["LTX_TRANSFORMER_REFERENCE_LTX23"] != nil))
    func forwardMatchesUpstreamWithCrossAttentionAdaLN() throws {
        try compare(referencePath: Self.referencePathLTX23!, crossAttentionAdaLN: true)
    }

    func compare(referencePath: String, crossAttentionAdaLN: Bool) throws {
        let (tensors, _) = try MLX.loadArraysAndMetadata(
            url: URL(fileURLWithPath: referencePath))

        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix("weight.") {
            weights[String(key.dropFirst("weight.".count))] = value.asType(.float32)
        }
        #expect(!weights.isEmpty, "reference carries no weights")

        let model = LTXTransformer(
            config: Self.config(crossAttentionAdaLN: crossAttentionAdaLN), ropeType: .split)
        // The reference's keys are the checkpoint's, so they need the same
        // rewrites a real load applies (`to_out.0` → `to_out`, `ff.net.*`,
        // the AdaLN timestep embedder flattening).
        let mapped = LTXWeightLoader.mapTransformerWeights(weights)
        // Deliberately *not* through `applyTransformerWeights`: that converts
        // float32 to bfloat16 the way a real load does, and bf16 rounding through
        // the block stack is ~1e-2 relative — enough to hide any real mismatch.
        // The keys it would check are verified below instead.
        let declared = Set(model.parameters().flattened().map { $0.0 })
        let unexpected = mapped.keys.filter { !declared.contains($0) }.sorted()
        #expect(unexpected.isEmpty, "reference keys with no parameter: \(unexpected)")
        let missing = declared.filter { mapped[$0] == nil }.sorted()
            // Affine-free block norms carry no checkpoint weight, and the keyframe
            // marker is absent from a model built without it — zero, hence a no-op.
            .filter { !$0.hasSuffix(".norm1.weight") && !$0.hasSuffix(".norm2.weight")
                      && !$0.hasSuffix(".norm3.weight") && $0 != "keyframes_abs_pos_embedding" }
        #expect(missing.isEmpty, "parameters the reference does not feed: \(missing)")
        _ = model.update(parameters: ModuleParameters.unflattened(mapped))
        MLX.eval(model.parameters())

        // Everything runs in float32 here: this is a correctness check, and
        // bfloat16 rounding would hide a real mismatch under its own noise.
        let latent = try #require(tensors["input.latent"]).asType(.float32)
        let context = try #require(tensors["input.context"]).asType(.float32)
        let expected = try #require(tensors["output.velocity"]).asType(.float32)

        let velocity = model(
            latent: latent,
            context: context,
            timesteps: MLXArray([Self.sigma]),
            contextMask: nil,
            latentShape: Self.shape
        ).asType(.float32)
        MLX.eval(velocity)

        #expect(velocity.shape == expected.shape)
        let absolute = MLX.abs(velocity - expected).max().item(Float.self)
        let scale = MLX.abs(expected).mean().item(Float.self)
        let relative = absolute / max(scale, 1e-8)
        print(String(format: "PARITY transformer: max|Δ| %.3e, mean|ref| %.4f, relative %.3e",
                     absolute, scale, relative))
        #expect(relative < 2e-4, "max|Δ| \(absolute) against mean |ref| \(scale)")
    }
}
