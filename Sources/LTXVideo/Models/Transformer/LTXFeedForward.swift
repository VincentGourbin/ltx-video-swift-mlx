// LTXFeedForward.swift - Feed-Forward Networks for LTX-2 Transformer
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - GELU Approximation

/// GELU activation with tanh approximation
///
/// This is the fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
public func geluApprox(_ x: MLXArray) -> MLXArray {
    return MLXNN.geluApproximate(x)
}

/// Linear layer followed by GELU (tanh approximation)
public class GELUApprox: Module, UnaryLayer {
    @ModuleInfo var proj: Linear

    public init(dimIn: Int, dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXNN.geluApproximate(proj(x))
    }
}

// MARK: - Feed-Forward Network

/// Feed-forward network with GELU activation
///
/// Architecture: Linear -> GELU -> Linear
public class LTXFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "project_in") var projectIn: GELUApprox
    @ModuleInfo(key: "project_out") var projectOut: Linear

    public init(dim: Int, dimOut: Int? = nil, mult: Int = 4) {
        let innerDim = dim * mult
        let outputDim = dimOut ?? dim

        self._projectIn.wrappedValue = GELUApprox(dimIn: dim, dimOut: innerDim)
        self._projectOut.wrappedValue = Linear(innerDim, outputDim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = projectIn(x)
        out = projectOut(out)
        return out
    }
}

// MARK: - SwiGLU Feed-Forward

/// SwiGLU feed-forward network (alternative to standard FFN)
///
/// Architecture: x -> Linear_gate * SiLU(Linear_up) -> Linear_down
public class SwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "w_up") var wUp: Linear
    @ModuleInfo(key: "w_gate") var wGate: Linear
    @ModuleInfo(key: "w_down") var wDown: Linear

    public init(dim: Int, dimOut: Int? = nil, mult: Int = 4) {
        let innerDim = dim * mult
        let outputDim = dimOut ?? dim

        self._wUp.wrappedValue = Linear(dim, innerDim, bias: false)
        self._wGate.wrappedValue = Linear(dim, innerDim, bias: false)
        self._wDown.wrappedValue = Linear(innerDim, outputDim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SiLU(gate) * up
        let gate = MLXNN.silu(wGate(x))
        let up = wUp(x)
        let fused = gate * up
        return wDown(fused)
    }
}
