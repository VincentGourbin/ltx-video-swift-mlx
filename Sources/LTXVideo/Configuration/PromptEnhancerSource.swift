// PromptEnhancerSource.swift - Where the LTX-2.5 prompt enhancer's weights come from
// Copyright 2025

import Foundation

/// Precision of the managed Gemma 4 E2B-it prompt enhancer.
///
/// The enhancer is a *separate* generative model from the checkpoint's bundled
/// encoder: on LTX-2.5 the bundled Gemma 4 is encode-only (vestigial LM head,
/// measured — `docs/knowledge`), so enhancement runs on a small E2B-it instruct
/// model, as upstream's `--prompt-enhancer-gemma-root` does.
public enum PromptEnhancerPrecision: String, Sendable, CaseIterable {
    /// `mlx-community/gemma-4-e2b-it-bf16` (~10.2 GB). What the reference space
    /// runs, and the default here for that reason.
    case bf16

    /// `mlx-community/gemma-4-e2b-it-6bit` (~4.7 GB) — 5.5 GB less on disk.
    ///
    /// The checkpoint carries a standard MLX `quantization` block
    /// (`bits: 6, group_size: 64, mode: affine`) which the loader applies on its
    /// own, so it loads through exactly the same path as bf16.
    ///
    /// **Quality is not measured.** The 4-bit E2B was tried and degraded
    /// instruction following; 6-bit sits between that and the reference bf16 and
    /// nobody has run it against the enhancer bench. Opt in, compare on your own
    /// prompts (`docs/examples/ltx-2.5/enhancer-bench`), and keep bf16 if the
    /// captions drift.
    case sixBit = "6bit"

    /// HuggingFace repository holding this precision.
    public var repoID: String {
        switch self {
        case .bf16: return "mlx-community/gemma-4-e2b-it-bf16"
        case .sixBit: return "mlx-community/gemma-4-e2b-it-6bit"
        }
    }

    /// Approximate download size, for the message shown before a multi-GB pull.
    public var approximateSizeGB: Double {
        switch self {
        case .bf16: return 10.2
        case .sixBit: return 4.7
        }
    }
}

/// Where ``LTXPipeline`` gets the LTX-2.5 prompt enhancer.
///
/// Two ways to avoid a second copy of Gemma 4 on disk: pull a smaller managed
/// one, or point at weights the host application already ships.
///
/// ```swift
/// // Reuse an E2B-it checkpoint the app already has — nothing is downloaded.
/// let pipeline = LTXPipeline(model: .v25Distilled,
///                            promptEnhancer: .localRoot(myGemmaDir.path))
/// ```
public enum PromptEnhancerSource: Sendable, Equatable {
    /// Download and cache `mlx-community/gemma-4-e2b-it-<precision>` under the
    /// models directory, so `--models-dir` routes it like every other model.
    case managed(PromptEnhancerPrecision)

    /// A Gemma 4 E2B-it checkpoint directory the caller already has on disk.
    ///
    /// Nothing is downloaded and nothing is validated beyond the directory
    /// existing and holding a `config.json` — the loader reports what it cannot
    /// read. Any precision works: a `quantization` block in the config is
    /// applied by the loader.
    case localRoot(String)

    /// `mlx-community/gemma-4-e2b-it-bf16`, matching the reference space.
    public static let `default` = PromptEnhancerSource.managed(.bf16)
}
