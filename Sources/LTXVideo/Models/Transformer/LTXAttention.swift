// LTXAttention.swift - Attention Mechanisms for LTX-2 Transformer
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - RMS Normalization

/// Root Mean Square Layer Normalization
public class RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    public init(dims: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dims])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Apply RMS normalization without learnable weight
public func rmsNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let weight = MLXArray.ones([x.dim(-1)])
    return MLXFast.rmsNorm(x, weight: weight, eps: eps)
}

// MARK: - Scaled Dot-Product Attention

/// Scaled dot-product attention using optimized MLX implementation
///
/// Uses MLXFast.scaledDotProductAttention which is memory-efficient
/// (Flash Attention style) and runs optimized Metal kernels.
public func scaledDotProductAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray? = nil,
    scale: Float? = nil
) -> MLXArray {
    let s = scale ?? (1.0 / sqrt(Float(query.dim(-1))))
    return MLXFast.scaledDotProductAttention(
        queries: query,
        keys: key,
        values: value,
        scale: s,
        mask: mask
    )
}

// MARK: - Attention Core

/// Attention core computation
private func attentionCore(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    heads: Int,
    dimHead: Int,
    mask: MLXArray? = nil
) -> MLXArray {
    let b = q.dim(0)
    let tQ = q.dim(1)
    let tK = k.dim(1)

    // Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
    var qReshaped = q.reshaped([b, tQ, heads, dimHead]).transposed(0, 2, 1, 3)
    var kReshaped = k.reshaped([b, tK, heads, dimHead]).transposed(0, 2, 1, 3)
    var vReshaped = v.reshaped([b, tK, heads, dimHead]).transposed(0, 2, 1, 3)

    // Handle mask dimensions
    var attnMask = mask
    if let m = mask {
        if m.ndim == 2 {
            attnMask = m.reshaped([1, 1, m.dim(0), m.dim(1)])
        } else if m.ndim == 3 {
            attnMask = MLX.expandedDimensions(m, axis: 1)
        }
        // Ensure mask dtype matches query dtype
        attnMask = attnMask?.asType(qReshaped.dtype)
    }

    // Compute attention using Flash Attention
    let scale = 1.0 / sqrt(Float(dimHead))
    let out = MLXFast.scaledDotProductAttention(
        queries: qReshaped,
        keys: kReshaped,
        values: vReshaped,
        scale: scale,
        mask: attnMask
    )

    // Reshape back: (B, H, T, D) -> (B, T, H*D)
    return out.transposed(0, 2, 1, 3).reshaped([b, tQ, heads * dimHead])
}

// MARK: - Attention Module

/// Multi-head attention with RMSNorm on Q/K and optional RoPE
///
/// This attention module follows the LTX-2 architecture:
/// - RMSNorm applied to Q and K before attention
/// - RoPE applied to Q and K (if position embeddings provided)
/// - Standard scaled dot-product attention
public class LTXAttention: Module {
    let ropeType: LTXRopeType
    let heads: Int
    let dimHead: Int

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(
        queryDim: Int,
        contextDim: Int? = nil,
        heads: Int = 8,
        dimHead: Int = 64,
        normEps: Float = 1e-6,
        ropeType: LTXRopeType = .split
    ) {
        self.ropeType = ropeType
        self.heads = heads
        self.dimHead = dimHead
        let innerDim = dimHead * heads

        let ctxDim = contextDim ?? queryDim

        // RMSNorm for Q and K
        self._qNorm.wrappedValue = RMSNorm(dims: innerDim, eps: normEps)
        self._kNorm.wrappedValue = RMSNorm(dims: innerDim, eps: normEps)

        // Linear projections
        self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(ctxDim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(ctxDim, innerDim, bias: true)

        // Output projection
        self._toOut.wrappedValue = Linear(innerDim, queryDim, bias: true)
    }

    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray? = nil,
        mask: MLXArray? = nil,
        pe: (cos: MLXArray, sin: MLXArray)? = nil,
        kPe: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        // Project to Q, K, V
        var q = toQ(x)
        let ctx = context ?? x
        var k = toK(ctx)
        let v = toV(ctx)

        // Apply RMSNorm to Q and K
        q = qNorm(q)
        k = kNorm(k)

        // Apply RoPE if position embeddings provided
        if let posEmb = pe {
            q = applyRotaryEmb(q, freqsCis: posEmb, ropeType: ropeType)
            let keyPosEmb = kPe ?? posEmb
            k = applyRotaryEmb(k, freqsCis: keyPosEmb, ropeType: ropeType)
        }

        // Compute attention
        let out = attentionCore(q: q, k: k, v: v, heads: heads, dimHead: dimHead, mask: mask)

        // Output projection
        return toOut(out)
    }
}

// MARK: - Self-Attention

/// Self-attention layer (convenience wrapper)
public class SelfAttention: Module {
    @ModuleInfo var attn: LTXAttention

    public init(
        dim: Int,
        heads: Int = 8,
        dimHead: Int = 64,
        normEps: Float = 1e-6,
        ropeType: LTXRopeType = .split
    ) {
        self._attn.wrappedValue = LTXAttention(
            queryDim: dim,
            contextDim: dim,
            heads: heads,
            dimHead: dimHead,
            normEps: normEps,
            ropeType: ropeType
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        pe: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        return attn(x, context: nil, mask: mask, pe: pe)
    }
}

// MARK: - Cross-Attention

/// Cross-attention layer (convenience wrapper)
public class CrossAttention: Module {
    @ModuleInfo var attn: LTXAttention

    public init(
        queryDim: Int,
        contextDim: Int,
        heads: Int = 8,
        dimHead: Int = 64,
        normEps: Float = 1e-6
    ) {
        // Cross-attention typically doesn't use RoPE
        self._attn.wrappedValue = LTXAttention(
            queryDim: queryDim,
            contextDim: contextDim,
            heads: heads,
            dimHead: dimHead,
            normEps: normEps,
            ropeType: .split
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        return attn(x, context: context, mask: mask, pe: nil)
    }
}
