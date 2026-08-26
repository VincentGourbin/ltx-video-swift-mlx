// PromptEnhancerOptions.swift - Shared CLI flags selecting the LTX-2.5 prompt enhancer
// Copyright 2025

import ArgumentParser
import LTXVideo

extension PromptEnhancerPrecision: ExpressibleByArgument {
    public static var allValueStrings: [String] { allCases.map(\.rawValue) }
}

/// Flags shared by every command that offers `--enhance-prompt`.
///
/// LTX-2.5 enhancement runs on a *separate* Gemma 4 E2B-it, because the bundled
/// encoder is encode-only. That is a second multi-gigabyte Gemma on disk next to
/// the 26 GB in-checkpoint encoder, which these flags exist to avoid: pull a
/// smaller one, or reuse a checkpoint already on the machine.
struct PromptEnhancerOptions: ParsableArguments {
    @Option(
        name: .long,
        help: ArgumentHelp(
            "LTX-2.5 only. Directory holding a Gemma 4 E2B-it checkpoint to use as the prompt "
            + "enhancer, instead of downloading one. Any precision — a quantization block in "
            + "its config.json is applied on load. Mirrors upstream's "
            + "--prompt-enhancer-gemma-root.",
            valueName: "dir"))
    var promptEnhancerRoot: String?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "LTX-2.5 only. Precision of the downloaded prompt enhancer: bf16 (~10.2GB, what the "
            + "reference space runs, default) or 6bit (~4.7GB, 5.5GB less on disk, quality "
            + "unmeasured).",
            valueName: "bf16|6bit"))
    var promptEnhancerPrecision: PromptEnhancerPrecision?

    init() {}

    /// Refuse the combination that would silently ignore one of the two flags.
    ///
    /// Here rather than in ``resolve()`` so it fires before the command prints a
    /// banner and starts resolving checkpoints — ArgumentParser runs `validate()`
    /// on option groups ahead of `run()`.
    func validate() throws {
        guard promptEnhancerRoot == nil || promptEnhancerPrecision == nil else {
            throw ValidationError(
                "--prompt-enhancer-precision applies to the downloaded enhancer and means "
                + "nothing alongside --prompt-enhancer-root, whose precision is whatever that "
                + "checkpoint already is. Pass one or the other.")
        }
    }

    /// Turn the flags into a source.
    func resolve() throws -> PromptEnhancerSource {
        try validate()
        if let root = promptEnhancerRoot { return .localRoot(root) }
        return .managed(promptEnhancerPrecision ?? .bf16)
    }
}

