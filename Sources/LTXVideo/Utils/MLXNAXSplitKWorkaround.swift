// MLXNAXSplitKWorkaround.swift - Gated M5 workaround for an mlx NAX split-K GEMM bug
// Copyright 2026

import Foundation
@preconcurrency import MLX

/// Detects the exact hardware/OS combination that hits
/// [ml-explore/mlx#3797](https://github.com/ml-explore/mlx/issues/3797): the
/// Metal NAX split-K GEMM kernel (`steel_gemm_splitk_axpby_nax`) is templated
/// on its split-K accumulator's dtype — hardcoded `float32`
/// (`mlx/backend/metal/matmul.cpp`'s `C_split`) — instead of the actual input
/// dtype (`get_type_string(out.dtype())` should read `in.dtype()`,
/// `mlx/backend/metal/jit_kernels.cpp`). For a bf16 input the kernel is
/// templated for float32 but fed bf16-laid-out data, so it misreads the
/// buffer: connector output collapses to zero, then NaN.
///
/// Fixed upstream by [mlx#3810](https://github.com/ml-explore/mlx/pull/3810)
/// (2026-07-07), but no mlx-swift tag vendors it yet — bumping to `main`
/// changes generation output on unaffected hardware too (measured ~20-28%
/// relative on a plain bf16 generation on an M3 Max, unrelated to this bug;
/// see `docs/knowledge/pitfalls/`). Until a tagged fix lands, this workaround
/// casts the one exposed op (the text connector's feed-forward down
/// projection — `[1024, 4096] @ [4096, 16384]`, `M·N` sits exactly on the
/// `2048²` dispatch threshold) to float32, matching the accumulator's
/// hardcoded dtype and sidestepping the mismatch — but **only** on hardware
/// this bug can actually hit, leaving every other machine's bf16 path (and
/// the numbers `ConnectorParityTests` was measured against) untouched.
///
/// Mirrors `is_nax_available()` (`mlx/backend/metal/device.h`) exactly,
/// using the same input (`MTLDevice.architecture.name`, already surfaced
/// publicly as `MLX.GPU.deviceInfo().architecture`) and the same parsing:
/// the two digits before the architecture string's last character are the
/// GPU generation; that last character is `'p'` for phone-class silicon
/// (threshold 18) and anything else for base/pro/max/ultra (threshold 17).
enum MLXNAXSplitKWorkaround {
    /// Whether this process is running on hardware/OS where the NAX
    /// split-K dispatch condition (`mlx#3797`) can actually be reached.
    ///
    /// Computed once per process — the underlying GPU architecture and OS
    /// version cannot change at runtime.
    static let isAffectedHardware: Bool = computeIsAffectedHardware()

    private static func computeIsAffectedHardware() -> Bool {
        // Manual override: "1"/"true" forces the workaround on (useful to
        // test the f32 path itself on hardware this bug doesn't hit, or if
        // detection ever misses a future GPU architecture string), "0"/
        // "false" forces it off. Unset falls through to auto-detection.
        if let override = ProcessInfo.processInfo.environment["LTX_NAX_WORKAROUND"] {
            let forced = ["1", "true", "yes"].contains(override.lowercased())
            LTXDebug.log("[NAXWorkaround] LTX_NAX_WORKAROUND=\(override) → forced affected=\(forced)")
            return forced
        }

        guard #available(macOS 26.2, *) else { return false }

        let arch = Array(MLX.GPU.deviceInfo().architecture)
        guard arch.count >= 3 else { return false }

        // Same clamp-to-0-on-non-digit behavior as mlx's own parsing
        // (`ag_tens = (ag_tens < 10 && ag_tens >= 0) ? ag_tens : 0`), so a
        // string this repo hasn't seen yet degrades to "not affected"
        // rather than crashing or throwing.
        func digit(_ c: Character) -> Int {
            guard let v = c.wholeNumberValue, v >= 0, v < 10 else { return 0 }
            return v
        }

        let tens = digit(arch[arch.count - 3])
        let ones = digit(arch[arch.count - 2])
        let gen = tens * 10 + ones
        let threshold = (arch.last == "p") ? 18 : 17
        let affected = gen >= threshold
        LTXDebug.log(
            "[NAXWorkaround] architecture=\(String(arch)) gen=\(gen) threshold=\(threshold) "
            + "affected=\(affected)")
        return affected
    }
}
