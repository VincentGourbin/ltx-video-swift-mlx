// LTXRoPE.swift - 3D Rotary Position Embeddings for LTX-2
// Copyright 2025

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RoPE Type

/// RoPE implementation variants
enum LTXRopeType: String, Sendable {
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
func applyRotaryEmb(
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
    // Cast to float32 for rotation math (matching Python apply_interleaved_rotary_emb)
    let inputDtype = input.dtype
    let inputF32 = input.asType(.float32)
    let cosF32 = cosFreqs.asType(.float32)
    let sinF32 = sinFreqs.asType(.float32)

    let shape = inputF32.shape
    let dim = shape[shape.count - 1]

    // Reshape to pair dimensions: (..., dim) -> (..., dim/2, 2)
    var reshapeShape = Array(shape.dropLast())
    reshapeShape.append(dim / 2)
    reshapeShape.append(2)

    let tDup = inputF32.reshaped(reshapeShape)

    // Split into t1, t2
    let t1 = tDup[.ellipsis, 0]  // Even indices
    let t2 = tDup[.ellipsis, 1]  // Odd indices

    // Compute rotated version: (-t2, t1)
    let tRot = MLX.stacked([-t2, t1], axis: -1)
    let inputRot = tRot.reshaped(shape)

    // Apply rotation in float32: x * cos + x_rot * sin
    let result = inputF32 * cosF32 + inputRot * sinF32

    // Cast back to original dtype
    return result.asType(inputDtype)
}

/// Apply split rotary embeddings
///
/// The split format divides the dimension in half: first_half rotates with second_half.
private func applySplitRotaryEmb(
    _ input: MLXArray,
    cosFreqs: MLXArray,
    sinFreqs: MLXArray
) -> MLXArray {
    // Cast to float32 for rotation math (matching Python _apply_split_rope)
    let inputDtype = input.dtype
    var inputTensor = input.asType(.float32)
    let cosF32 = cosFreqs.asType(.float32)
    let sinF32 = sinFreqs.asType(.float32)

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
        inputTensor = inputTensor.reshaped([b, t, h, -1])
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

    // Apply rotation in float32
    // first_half_out = first_half * cos - second_half * sin
    // second_half_out = second_half * cos + first_half * sin
    let firstHalfOut = firstHalf * cosF32 - secondHalf * sinF32
    let secondHalfOut = secondHalf * cosF32 + firstHalf * sinF32

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

    // Cast back to original dtype
    return output.asType(inputDtype)
}

// MARK: - Frequency Generation

/// Generate frequency grid for RoPE
///
/// - Parameters:
///   - theta: Base theta value for frequencies
///   - maxPosCount: Number of position dimensions
///   - innerDim: Inner dimension size
/// - Returns: Frequency indices array
func generateFreqGrid(
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
func getFractionalPositions(
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
func generateFreqs(
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
func splitFreqsCis(
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
func interleavedFreqsCis(
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
/// Uses double precision (Float64) for frequency computation to match
/// Python's `double_precision_rope=True` behavior. This is critical for
/// numerical accuracy — float32 cos/sin values diverge from float64 by
/// enough to cause visible quality degradation over 48 transformer blocks
/// and 40 denoising steps.
///
/// - Parameters:
///   - indicesGrid: Position indices grid, shape (B, n_dims, T)
///   - dim: Dimension of the embedding
///   - theta: Base theta for frequency computation
///   - maxPos: Maximum positions per dimension [time, height, width]
///   - numAttentionHeads: Number of attention heads
///   - ropeType: Type of RoPE (INTERLEAVED or SPLIT)
/// - Returns: Tuple of (cos_freqs, sin_freqs) in float32
func precomputeFreqsCis(
    indicesGrid: MLXArray,
    dim: Int,
    theta: Float = 10000.0,
    maxPos: [Int]? = nil,
    numAttentionHeads: Int = 32,
    ropeType: LTXRopeType = .split,
    doublePrecision: Bool = false
) -> (cos: MLXArray, sin: MLXArray) {
    let maxPositions = maxPos ?? [20, 2048, 2048]

    // Double precision path: matches Python connector's numpy float64 computation.
    // The connector's _precompute_freqs_cis explicitly uses np.float64 for all
    // intermediate calculations, which matters for high-frequency components.
    if doublePrecision {
        return precomputeFreqsCisDoublePrecision(
            indicesGrid: indicesGrid,
            dim: dim,
            theta: Double(theta),
            maxPos: maxPositions,
            numAttentionHeads: numAttentionHeads,
            ropeType: ropeType
        )
    }

    // Float32 GPU computation: matches Python transformer's actual behavior.
    // NOTE: Python's transformer uses float32 MLX ops (not numpy float64).
    let indices = generateFreqGrid(theta: theta, maxPosCount: indicesGrid.dim(1), innerDim: dim)
    let freqs = generateFreqs(indices: indices, indicesGrid: indicesGrid, maxPos: maxPositions)

    switch ropeType {
    case .split:
        let expectedFreqs = dim / 2
        let currentFreqs = freqs.dim(-1)
        let padSize = max(0, expectedFreqs - currentFreqs)
        return splitFreqsCis(freqs: freqs, padSize: padSize, numAttentionHeads: numAttentionHeads)
    case .interleaved:
        let nElem = 2 * indicesGrid.dim(1)
        let padSize = dim % nElem
        return interleavedFreqsCis(freqs: freqs, padSize: padSize)
    }
}

/// Double-precision RoPE frequency computation on CPU
///
/// Matches Python `_precompute_freqs_cis_double_precision` exactly:
/// 1. Convert position grid to Float64 (via numpy in Python, via Swift Double here)
/// 2. Compute frequency indices in Float64
/// 3. Compute cos/sin in Float64
/// 4. Convert final result to Float32 for GPU
private func precomputeFreqsCisDoublePrecision(
    indicesGrid: MLXArray,
    dim: Int,
    theta: Double,
    maxPos: [Int],
    numAttentionHeads: Int,
    ropeType: LTXRopeType
) -> (cos: MLXArray, sin: MLXArray) {
    // Extract position grid to CPU as Float64
    // indicesGrid shape: (B, n_dims, T)
    let gridF32 = indicesGrid.asType(.float32)
    MLX.eval(gridF32)

    let batchSize = gridF32.dim(0)
    let nPosDims = gridF32.dim(1)
    let seqLen = gridF32.dim(2)
    let nElem = 2 * nPosDims

    // 1. Generate frequency indices in Float64
    let logStart = Foundation.log(1.0) / Foundation.log(theta)
    let logEnd = Foundation.log(theta) / Foundation.log(theta)  // = 1.0
    let numIndices = max(1, dim / nElem)

    var indicesF64 = [Double](repeating: 0, count: numIndices)
    for i in 0..<numIndices {
        let t = numIndices > 1
            ? logStart + (logEnd - logStart) * Double(i) / Double(numIndices - 1)
            : logStart
        indicesF64[i] = Foundation.pow(theta, t) * (Double.pi / 2.0)
    }

    // 2. Extract grid values to CPU Double arrays
    // Grid shape: (B, n_dims, T) → we need fractional positions (B, T, n_dims)
    var gridValues = [[Double]](repeating: [Double](repeating: 0, count: seqLen), count: nPosDims)
    for d in 0..<nPosDims {
        let slice = gridF32[0, d, 0...]  // (T,) — batch 0 (all batches are identical)
        MLX.eval(slice)
        let flatSlice = slice.flattened()
        MLX.eval(flatSlice)
        for t in 0..<seqLen {
            gridValues[d][t] = Double(flatSlice[t].item(Float.self))
        }
    }

    // 3. Compute fractional positions in Float64
    // frac[t][d] = gridValues[d][t] / maxPos[d]
    // scaled[t][d] = frac * 2 - 1
    var scaledPositions = [[Double]](repeating: [Double](repeating: 0, count: nPosDims), count: seqLen)
    for t in 0..<seqLen {
        for d in 0..<nPosDims {
            let frac = gridValues[d][t] / Double(maxPos[d])
            scaledPositions[t][d] = frac * 2.0 - 1.0
        }
    }

    // 4. Compute frequencies: freqs[t] = flatten(indices * scaledPositions[t])
    // Result shape: (T, numIndices * nPosDims)
    let freqDim = numIndices * nPosDims
    var freqsF64 = [Double](repeating: 0, count: seqLen * freqDim)
    for t in 0..<seqLen {
        for fi in 0..<numIndices {
            for d in 0..<nPosDims {
                // Match Python's transpose: (T, n_dims, n_freq) → (T, n_freq, n_dims)
                let outIdx = t * freqDim + fi * nPosDims + d
                freqsF64[outIdx] = indicesF64[fi] * scaledPositions[t][d]
            }
        }
    }

    // 5. Compute cos/sin in Float64
    let cosF64 = freqsF64.map { Foundation.cos($0) }
    let sinF64 = freqsF64.map { Foundation.sin($0) }

    // 6. Handle RoPE type-specific processing
    switch ropeType {
    case .split:
        let expectedFreqs = dim / 2
        let padSize = max(0, expectedFreqs - freqDim)

        // Pad with 1s for cos, 0s for sin (identity transform)
        // Final shape per token: padSize + freqDim = expectedFreqs = dim/2
        let totalFreqsPerToken = padSize + freqDim

        // Build padded arrays: (B, T, totalFreqsPerToken) → all batches identical
        var cosPadded = [Float](repeating: 0, count: batchSize * seqLen * totalFreqsPerToken)
        var sinPadded = [Float](repeating: 0, count: batchSize * seqLen * totalFreqsPerToken)

        for b in 0..<batchSize {
            for t in 0..<seqLen {
                let baseOut = (b * seqLen + t) * totalFreqsPerToken
                let baseIn = t * freqDim
                // Padding (cos=1, sin=0)
                for p in 0..<padSize {
                    cosPadded[baseOut + p] = 1.0
                    sinPadded[baseOut + p] = 0.0
                }
                // Frequency values
                for f in 0..<freqDim {
                    cosPadded[baseOut + padSize + f] = Float(cosF64[baseIn + f])
                    sinPadded[baseOut + padSize + f] = Float(sinF64[baseIn + f])
                }
            }
        }

        // Convert to MLXArray: (B, T, totalFreqsPerToken)
        var cosArray = MLXArray(cosPadded, [batchSize, seqLen, totalFreqsPerToken])
        var sinArray = MLXArray(sinPadded, [batchSize, seqLen, totalFreqsPerToken])

        // Reshape for multi-head: (B, T, H, D//2//H) → (B, H, T, D//2//H)
        let headDim = totalFreqsPerToken / numAttentionHeads
        cosArray = cosArray.reshaped([batchSize, seqLen, numAttentionHeads, headDim])
        sinArray = sinArray.reshaped([batchSize, seqLen, numAttentionHeads, headDim])
        cosArray = cosArray.transposed(0, 2, 1, 3)
        sinArray = sinArray.transposed(0, 2, 1, 3)

        return (cosArray, sinArray)

    case .interleaved:
        let padSize = dim % nElem

        // Repeat interleave: each freq value appears twice
        let repeatedDim = freqDim * 2
        let totalDim = repeatedDim + padSize

        var cosFinal = [Float](repeating: 0, count: batchSize * seqLen * totalDim)
        var sinFinal = [Float](repeating: 0, count: batchSize * seqLen * totalDim)

        for b in 0..<batchSize {
            for t in 0..<seqLen {
                let baseOut = (b * seqLen + t) * totalDim
                let baseIn = t * freqDim
                // Padding (cos=1, sin=0) at the beginning
                for p in 0..<padSize {
                    cosFinal[baseOut + p] = 1.0
                    sinFinal[baseOut + p] = 0.0
                }
                // Repeated frequency values
                for f in 0..<freqDim {
                    let c = Float(cosF64[baseIn + f])
                    let s = Float(sinF64[baseIn + f])
                    cosFinal[baseOut + padSize + f * 2] = c
                    cosFinal[baseOut + padSize + f * 2 + 1] = c
                    sinFinal[baseOut + padSize + f * 2] = s
                    sinFinal[baseOut + padSize + f * 2 + 1] = s
                }
            }
        }

        let cosArray = MLXArray(cosFinal, [batchSize, seqLen, totalDim])
        let sinArray = MLXArray(sinFinal, [batchSize, seqLen, totalDim])
        return (cosArray, sinArray)
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
func createPositionGrid(
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
