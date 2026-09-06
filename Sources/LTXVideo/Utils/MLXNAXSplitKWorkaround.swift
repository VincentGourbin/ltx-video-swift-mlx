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
/// see `docs/knowledge/decisions/mlx-swift-main-bump-rejected-2026-09.md`).
/// Until a tagged fix lands, this workaround
/// casts the one exposed op (the text connector's feed-forward down
/// projection — `[1024, 16384] @ [16384, 4096]`, `M·N` sits exactly on the
/// `2048²` dispatch threshold) to float32, matching the accumulator's
/// hardcoded dtype and sidestepping the mismatch — but **only** on hardware
/// this bug can actually hit, leaving every other machine's bf16 path (and
/// the numbers `ConnectorParityTests` was measured against) untouched.
///
/// Mirrors `is_nax_available()` (`mlx/backend/metal/device.h`) as closely as
/// this repo can from outside mlx's C++ core: the two digits before the
/// architecture string's last character are the GPU generation; that last
/// character is `'p'` for phone-class silicon (threshold 18) and anything
/// else for base/pro/max/ultra (threshold 17). Upstream's `Device`
/// constructor (`mlx/backend/metal/device.cpp`) resolves the architecture
/// string from the `MLX_METAL_GPU_ARCH` environment variable first, falling
/// back to the real `MTLDevice.architecture.name` only when that's unset —
/// so this checks the same env var before falling back to
/// `MLX.GPU.deviceInfo().architecture` (which, unlike upstream's internal
/// `Device`, always queries the real hardware and has no override of its
/// own). Without that, a process that sets `MLX_METAL_GPU_ARCH` to simulate
/// a gen-17+/18+ architecture (mlx's own supported way to exercise this path
/// off real hardware) would make mlx's *actual* kernel dispatch decide
/// "affected" while this gate — reading only the real, older GPU — decided
/// "not affected", silently leaving the workaround off exactly when the real
/// dispatch needs it.
enum MLXNAXSplitKWorkaround {
    /// Whether this process should treat itself as running on hardware/OS
    /// where the NAX split-K dispatch condition (`mlx#3797`) can actually be
    /// reached.
    ///
    /// The `LTX_NAX_WORKAROUND` override is re-read on every access — it's a
    /// local testing knob (not part of upstream mlx's own state), and a
    /// `static let` here would freeze whatever value was read on first
    /// access for the rest of the process, defeating `setenv`/`unsetenv`
    /// toggling around individual tests (the same problem this codebase's
    /// `RuntimeBeacon.environmentEnabled` already solves the same way, for
    /// the same reason). The underlying hardware/OS auto-detection *is*
    /// memoized (`cachedHardwareDetection` below) — the real GPU generation
    /// and OS version genuinely cannot change within one process.
    static var isAffectedHardware: Bool {
        // Manual override: "1" forces the workaround on (useful to test the
        // f32 path itself on hardware this bug doesn't hit, or if detection
        // ever misses a future GPU architecture string), "0" forces it off.
        // Unset *or any other value* falls through to auto-detection —
        // deliberately not a "recognized on/off spellings, else off" parse:
        // a blank/typo'd override should not silently behave like an
        // explicit "0" when someone is trying to force this on to diagnose
        // the exact bug this workaround exists for.
        if let override = ProcessInfo.processInfo.environment["LTX_NAX_WORKAROUND"] {
            if override == "1" {
                if LTXDebug.isEnabled {
                    LTXDebug.log("[NAXWorkaround] LTX_NAX_WORKAROUND=1 → forced affected=true")
                }
                return true
            }
            if override == "0" {
                if LTXDebug.isEnabled {
                    LTXDebug.log("[NAXWorkaround] LTX_NAX_WORKAROUND=0 → forced affected=false")
                }
                return false
            }
        }
        return cachedHardwareDetection
    }

    private static let cachedHardwareDetection: Bool = computeIsAffectedHardware()

    private static func computeIsAffectedHardware() -> Bool {
        guard #available(macOS 26.2, *) else { return false }

        // Mirror mlx's own Device constructor precedence: MLX_METAL_GPU_ARCH
        // wins over the real hardware string when set.
        let archOverride = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"]
        let archString = (archOverride?.isEmpty == false)
            ? archOverride!
            : MLX.GPU.deviceInfo().architecture
        let arch = Array(archString)
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
        if LTXDebug.isEnabled {
            LTXDebug.log(
                "[NAXWorkaround] architecture=\(archString) gen=\(gen) threshold=\(threshold) "
                + "affected=\(affected)")
        }
        return affected
    }
}
