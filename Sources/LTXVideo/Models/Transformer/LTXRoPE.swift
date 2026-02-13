// LTXRoPE.swift - 3D Rotary Position Embeddings for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RoPE Type

/// RoPE implementation variants
public enum LTXRopeType: String, Sendable {
    /// Interleaved format: pairs adjacent dimensions (d0, d1), (d2, d3), ...
    case interleaved = "interleaved"

    /// Split format: divides dimension in half, first half rotates with second half
    case split = "split"
}

// MARK: - RoPE Application

/// Apply rotary position embeddings to input tensor
///
/// - Parameters:
///   - input: Input tensor to apply RoPE to
///   - freqsCis: Tuple of (cos_freqs, sin_freqs)
///   - ropeType: Type of RoPE implementation
/// - Returns: Tensor with rotary embeddings applied
public func applyRotaryEmb(
    _ input: MLXArray,
    freqsCis: (cos: MLXArray, sin: MLXArray),
    ropeType: LTXRopeType = .split
) -> MLXArray {
    switch ropeType {
    case .interleaved:
        return applyInterleavedRotaryEmb(input, cosFreqs: freqsCis.cos, sinFreqs: freqsCis.sin)
    case .split:
        return applySplitRotaryEmb(input, cosFreqs: freqsCis.cos, sinFreqs: freqsCis.sin)
    }
}

/// Apply interleaved rotary embeddings
///
/// The interleaved format pairs adjacent dimensions: (d0, d1), (d2, d3), ...
/// Rotation is applied to each pair.
private func applyInterleavedRotaryEmb(
    _ input: MLXArray,
    cosFreqs: MLXArray,
    sinFreqs: MLXArray
) -> MLXArray {
    let shape = input.shape
    let dim = shape[shape.count - 1]

    // Reshape to pair dimensions: (..., dim) -> (..., dim/2, 2)
    var reshapeShape = Array(shape.dropLast())
    reshapeShape.append(dim / 2)
    reshapeShape.append(2)

    let tDup = input.reshaped(reshapeShape)

    // Split into t1, t2
    let t1 = tDup[.ellipsis, 0]  // Even indices
    let t2 = tDup[.ellipsis, 1]  // Odd indices

    // Compute rotated version: (-t2, t1)
    let tRot = MLX.stacked([-t2, t1], axis: -1)
    let inputRot = tRot.reshaped(shape)

    // Apply rotation: x * cos + x_rot * sin
    return input * cosFreqs + inputRot * sinFreqs
}

/// Apply split rotary embeddings
///
/// The split format divides the dimension in half: first_half rotates with second_half.
private func applySplitRotaryEmb(
    _ input: MLXArray,
    cosFreqs: MLXArray,
    sinFreqs: MLXArray
) -> MLXArray {
    var inputTensor = input
    var needsReshape = false
    var originalB = 0
    var originalH = 0

    // Handle dimension mismatch - reshape if needed
    if input.ndim != 4 && cosFreqs.ndim == 4 {
        let b = cosFreqs.dim(0)
        let h = cosFreqs.dim(1)
        let t = cosFreqs.dim(2)

        originalB = b
        originalH = h

        // Reshape from (B, T, H*D) to (B, H, T, D)
        inputTensor = input.reshaped([b, t, h, -1])
        inputTensor = inputTensor.transposed(0, 2, 1, 3)
        needsReshape = true
    }

    // Split into two halves
    let dim = inputTensor.dim(-1)
    var splitShape = Array(inputTensor.shape.dropLast())
    splitShape.append(2)
    splitShape.append(dim / 2)

    let splitInput = inputTensor.reshaped(splitShape)

    let firstHalf = splitInput[.ellipsis, 0, 0...]  // Shape: (..., dim//2)
    let secondHalf = splitInput[.ellipsis, 1, 0...]  // Shape: (..., dim//2)

    // Apply rotation
    // first_half_out = first_half * cos - second_half * sin
    // second_half_out = second_half * cos + first_half * sin
    let firstHalfOut = firstHalf * cosFreqs - secondHalf * sinFreqs
    let secondHalfOut = secondHalf * cosFreqs + firstHalf * sinFreqs

    // Stack back together
    var output = MLX.stacked([firstHalfOut, secondHalfOut], axis: -2)

    // Reshape back to original dimension layout
    var outShape = Array(output.shape.dropLast(2))
    outShape.append(dim)
    output = output.reshaped(outShape)

    if needsReshape {
        let t = output.dim(2)
        let d = output.dim(3)
        // Reshape from (B, H, T, D) back to (B, T, H*D)
        output = output.transposed(0, 2, 1, 3)
        output = output.reshaped([originalB, t, originalH * d])
    }

    return output
}

// MARK: - Frequency Generation

/// Generate frequency grid for RoPE
///
/// - Parameters:
///   - theta: Base theta value for frequencies
///   - maxPosCount: Number of position dimensions
///   - innerDim: Inner dimension size
/// - Returns: Frequency indices array
public func generateFreqGrid(
    theta: Float,
    maxPosCount: Int,
    innerDim: Int
) -> MLXArray {
    let start: Float = 1.0
    let end = theta
    let nElem = 2 * maxPosCount

    // Generate logarithmically spaced indices
    let logStart = log(start) / log(theta)
    let logEnd = log(end) / log(theta)
    let numIndices = innerDim / nElem

    guard numIndices > 0 else {
        return MLXArray([Float]())
    }

    let linspace = MLX.linspace(Float32(logStart), Float32(logEnd), count: numIndices)
    var indices = MLX.pow(MLXArray(theta), linspace)
    indices = indices * Float32(Float.pi / 2)

    return indices.asType(.float32)
}

/// Convert position indices to fractional positions in [0, 1]
///
/// - Parameters:
///   - indicesGrid: Grid of position indices, shape (B, n_pos_dims, T)
///   - maxPos: Maximum position for each dimension
/// - Returns: Fractional positions, shape (B, T, n_pos_dims)
public func getFractionalPositions(
    indicesGrid: MLXArray,
    maxPos: [Int]
) -> MLXArray {
    let nPosDims = indicesGrid.dim(1)
    precondition(nPosDims == maxPos.count, "Number of position dimensions must match maxPos length")

    var fractional: [MLXArray] = []
    for i in 0..<nPosDims {
        let normalized = indicesGrid[0..., i, 0...] / Float32(maxPos[i])
        fractional.append(normalized)
    }

    // Stack along last dimension: (B, T, n_pos_dims)
    return MLX.stacked(fractional, axis: -1)
}

/// Generate frequencies from position indices
///
/// - Parameters:
///   - indices: Frequency indices
///   - indicesGrid: Position grid, shape (B, n_dims, T)
///   - maxPos: Maximum positions per dimension
/// - Returns: Frequencies array
public func generateFreqs(
    indices: MLXArray,
    indicesGrid: MLXArray,
    maxPos: [Int]
) -> MLXArray {
    // Get fractional positions: (B, T, n_dims)
    let fractionalPositions = getFractionalPositions(indicesGrid: indicesGrid, maxPos: maxPos)

    // Compute frequencies: scale fractional positions to [-1, 1] range
    var scaledPositions = fractionalPositions * 2 - 1  // (B, T, n_dims)
    scaledPositions = MLX.expandedDimensions(scaledPositions, axis: -1)  // (B, T, n_dims, 1)

    // indices shape: (n_freq,) -> broadcast to (1, 1, 1, n_freq)
    let indicesExp = indices.expandedDimensions(axes: [0, 1, 2])  // (1, 1, 1, n_freq)

    // freqs shape: (B, T, n_dims, n_freq)
    let freqs = indicesExp * scaledPositions

    // Transpose and flatten: (B, T, n_freq, n_dims) -> (B, T, n_freq * n_dims)
    let transposed = freqs.transposed(0, 1, 3, 2)
    let b = transposed.dim(0)
    let t = transposed.dim(1)
    return transposed.reshaped([b, t, -1])
}

// MARK: - Precompute Frequencies

/// Compute cos/sin frequencies for split RoPE format
///
/// - Parameters:
///   - freqs: Frequency array, shape (B, T, freq_dim)
///   - padSize: Padding size for dimensions that don't get RoPE
///   - numAttentionHeads: Number of attention heads
/// - Returns: Tuple of (cos_freq, sin_freq), each shape (B, H, T, D//2)
public func splitFreqsCis(
    freqs: MLXArray,
    padSize: Int,
    numAttentionHeads: Int
) -> (cos: MLXArray, sin: MLXArray) {
    var cosFreq = MLX.cos(freqs)
    var sinFreq = MLX.sin(freqs)

    if padSize > 0 {
        // Pad with 1s for cos and 0s for sin (identity transform)
        let cosPadding = MLX.ones([cosFreq.dim(0), cosFreq.dim(1), padSize])
        let sinPadding = MLX.zeros([sinFreq.dim(0), sinFreq.dim(1), padSize])

        cosFreq = MLX.concatenated([cosPadding, cosFreq], axis: -1)
        sinFreq = MLX.concatenated([sinPadding, sinFreq], axis: -1)
    }

    // Reshape for multi-head attention: (B, T, D) -> (B, H, T, D//H)
    let b = cosFreq.dim(0)
    let t = cosFreq.dim(1)

    cosFreq = cosFreq.reshaped([b, t, numAttentionHeads, -1])
    sinFreq = sinFreq.reshaped([b, t, numAttentionHeads, -1])

    // Transpose to (B, H, T, D//H)
    cosFreq = cosFreq.transposed(0, 2, 1, 3)
    sinFreq = sinFreq.transposed(0, 2, 1, 3)

    return (cosFreq, sinFreq)
}

/// Compute cos/sin frequencies for interleaved RoPE format
///
/// - Parameters:
///   - freqs: Frequency array, shape (B, T, freq_dim)
///   - padSize: Padding size
/// - Returns: Tuple of (cos_freq, sin_freq), each shape (B, T, dim)
public func interleavedFreqsCis(
    freqs: MLXArray,
    padSize: Int
) -> (cos: MLXArray, sin: MLXArray) {
    // Compute cos and sin, then repeat each value twice for interleaved format
    var cosFreq = MLX.cos(freqs)
    var sinFreq = MLX.sin(freqs)

    // Repeat interleave: each element appears twice
    cosFreq = MLX.repeated(cosFreq, count: 2, axis: -1)
    sinFreq = MLX.repeated(sinFreq, count: 2, axis: -1)

    if padSize > 0 {
        let cosPadding = MLX.ones([cosFreq.dim(0), cosFreq.dim(1), padSize])
        let sinPadding = MLX.zeros([sinFreq.dim(0), sinFreq.dim(1), padSize])

        cosFreq = MLX.concatenated([cosPadding, cosFreq], axis: -1)
        sinFreq = MLX.concatenated([sinPadding, sinFreq], axis: -1)
    }

    return (cosFreq, sinFreq)
}

/// Precompute cosine and sine frequencies for RoPE
///
/// - Parameters:
///   - indicesGrid: Position indices grid, shape (B, n_dims, T)
///   - dim: Dimension of the embedding
///   - theta: Base theta for frequency computation
///   - maxPos: Maximum positions per dimension [time, height, width]
///   - numAttentionHeads: Number of attention heads
///   - ropeType: Type of RoPE (INTERLEAVED or SPLIT)
/// - Returns: Tuple of (cos_freqs, sin_freqs)
public func precomputeFreqsCis(
    indicesGrid: MLXArray,
    dim: Int,
    theta: Float = 10000.0,
    maxPos: [Int]? = nil,
    numAttentionHeads: Int = 32,
    ropeType: LTXRopeType = .split
) -> (cos: MLXArray, sin: MLXArray) {
    let maxPositions = maxPos ?? [20, 2048, 2048]  // Default: [time, height, width]

    // Generate frequency indices
    let nPosDims = indicesGrid.dim(1)
    let indices = generateFreqGrid(theta: theta, maxPosCount: nPosDims, innerDim: dim)

    // Generate frequencies from positions
    let freqs = generateFreqs(indices: indices, indicesGrid: indicesGrid, maxPos: maxPositions)

    // Compute cos/sin based on RoPE type
    switch ropeType {
    case .split:
        let expectedFreqs = dim / 2
        let currentFreqs = freqs.dim(-1)
        let padSize = max(0, expectedFreqs - currentFreqs)
        return splitFreqsCis(freqs: freqs, padSize: padSize, numAttentionHeads: numAttentionHeads)

    case .interleaved:
        let nElem = 2 * nPosDims
        let padSize = dim % nElem
        return interleavedFreqsCis(freqs: freqs, padSize: padSize)
    }
}

// MARK: - Position Grid Creation

/// Create a 3D position grid for video tokens in pixel space
///
/// Converts latent-space indices to pixel-space middle coordinates following the Python
/// LTX-2-MLX pipeline: get_patch_grid_bounds → get_pixel_coords → causal_fix → /fps → middle
///
/// For patch_size=1, each latent position i maps to pixel bounds [i*scale, (i+1)*scale].
/// After causal fix (temporal only): bounds shift by (1 - temporalScale) and clamp to 0.
/// After FPS division (temporal only): temporal coords are divided by fps.
/// Final positions are the middle of each patch's pixel bounds.
///
/// - Parameters:
///   - batchSize: Batch size
///   - frames: Number of frames in latent space
///   - height: Height in latent space
///   - width: Width in latent space
///   - temporalScale: Temporal VAE scale factor (default: 8)
///   - spatialScale: Spatial VAE scale factor (default: 32)
///   - fps: Frames per second for temporal normalization (default: 24)
///   - causalFix: Whether to apply causal temporal fix (default: true)
/// - Returns: Position grid of shape (B, 3, T) where T = frames * height * width,
///            with pixel-space middle coordinates (temporal divided by fps)
public func createPositionGrid(
    batchSize: Int,
    frames: Int,
    height: Int,
    width: Int,
    temporalScale: Int = 8,
    spatialScale: Int = 32,
    fps: Float = 24.0,
    causalFix: Bool = true
) -> MLXArray {
    // GPU-based meshgrid using broadcasting (like Python's mx.meshgrid)

    // --- Temporal positions: pixel-space with causal fix and FPS division ---
    // For each frame i: start = i * temporalScale, end = (i+1) * temporalScale
    // Causal fix: shift by (1 - temporalScale), clamp to 0
    // Middle: (start + end) / 2, then divide by fps
    let tScale = Float(temporalScale)
    let tCoordValues: [Float] = (0..<frames).map { i in
        let fi = Float(i)
        var start = fi * tScale
        var end = (fi + 1) * tScale
        if causalFix {
            start = max(start + (1 - tScale), 0)
            end = max(end + (1 - tScale), 0)
        }
        return ((start + end) / 2.0) / fps
    }
    let tCoords = MLXArray(tCoordValues).reshaped([frames, 1, 1])

    // --- Spatial positions: pixel-space middle coordinates ---
    // For each position i: middle = i * spatialScale + spatialScale / 2
    let sScale = Float(spatialScale)
    let hCoordValues: [Float] = (0..<height).map { i in
        Float(i) * sScale + sScale / 2.0
    }
    let hCoords = MLXArray(hCoordValues).reshaped([1, height, 1])

    let wCoordValues: [Float] = (0..<width).map { i in
        Float(i) * sScale + sScale / 2.0
    }
    let wCoords = MLXArray(wCoordValues).reshaped([1, 1, width])

    // Broadcast to [frames, height, width] then flatten to [F*H*W]
    let tGrid = MLX.broadcast(tCoords, to: [frames, height, width]).flattened()
    let hGrid = MLX.broadcast(hCoords, to: [frames, height, width]).flattened()
    let wGrid = MLX.broadcast(wCoords, to: [frames, height, width]).flattened()

    // Stack positions: (3, T)
    let positions = MLX.stacked([tGrid, hGrid, wGrid], axis: 0)

    // Expand for batch: (B, 3, T)
    let totalTokens = frames * height * width
    let expanded = MLX.broadcast(
        positions.expandedDimensions(axis: 0),
        to: [batchSize, 3, totalTokens]
    )

    return expanded.asType(DType.float32)
}
