//
//  RoPENegativePositionTests.swift
//  ltx-video-swift-mlx
//
//  Validates that `precomputeFreqsCisDoublePrecision` produces the same
//  cos/sin values as Python `precompute_freqs_cis(..., generate_freq_grid_np)`
//  when positions include the negative range used by LipDub audio_ref tokens.
//
//  The Python expected values are produced by /tmp/rope_compare/compare_rope_negative.py
//  and persisted to /tmp/rope_compare/python_expected.json. If the JSON is
//  not present the test is skipped (no impact on CI).
//

import Testing
import Foundation
@preconcurrency import MLX
@testable import LTXVideo

@Suite("RoPE — negative position parity (audio_ref / LipDub)")
struct RoPENegativePositionTests {

    @Test func swiftMatchesPythonOnExtendedAudioPositions() throws {
        let jsonURL = URL(fileURLWithPath: "/tmp/rope_compare/python_expected.json")
        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            print("[SKIP] /tmp/rope_compare/python_expected.json missing — run compare_rope_negative.py first")
            return
        }

        let data = try Data(contentsOf: jsonURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let cfg = json["config"] as! [String: Any]
        let extMids = json["ext_mids"] as! [Double]
        let cosF64 = json["cos_f64"] as! [[Double]]   // (T, D//2)
        let sinF64 = json["sin_f64"] as! [[Double]]

        let audioInner = cfg["audio_inner"] as! Int
        let numHeads = cfg["num_heads"] as! Int
        let maxPos = (cfg["max_pos"] as! [Int])
        let theta = (cfg["theta"] as! NSNumber).floatValue

        let T = extMids.count
        let dHalf = cosF64[0].count
        print("[INFO] T=\(T)  D//2=\(dHalf)  audioInner=\(audioInner)  numHeads=\(numHeads)  maxPos=\(maxPos)")

        // Build the same 3D middle-position grid we use at runtime.
        let positionsArray = MLXArray(extMids.map { Float($0) }).reshaped([1, 1, T])

        // Run our precomputeFreqsCisDoublePrecision.
        let (cos, sin) = precomputeFreqsCis(
            indicesGrid: positionsArray,
            dim: audioInner,
            theta: theta,
            maxPos: maxPos,
            numAttentionHeads: numHeads,
            ropeType: .split,
            doublePrecision: true
        )
        // cos shape: (B, H, T, D//H/2) — head 0 should contain identical
        // content to all other heads in split RoPE (they share the same freqs).
        // Take head 0 for parity check.
        let cosHead0 = cos[0, 0, 0..., 0...]
        let sinHead0 = sin[0, 0, 0..., 0...]
        MLX.eval(cosHead0, sinHead0)

        let cosShape = cosHead0.shape
        let sinShape = sinHead0.shape
        print("[INFO] Swift cos head0 shape: \(cosShape) (expected [\(T), \(dHalf)])")
        #expect(cosShape == [T, dHalf])
        #expect(sinShape == [T, dHalf])

        // Compare element-wise.
        let cosArr = cosHead0.asArray(Float.self)
        let sinArr = sinHead0.asArray(Float.self)

        var maxCosDiff: Double = 0
        var maxSinDiff: Double = 0
        var maxDiffTok = -1
        var maxDiffFreq = -1
        var maxDiffMid: Double = 0
        for t in 0..<T {
            for d in 0..<dHalf {
                let cosSwift = Double(cosArr[t * dHalf + d])
                let sinSwift = Double(sinArr[t * dHalf + d])
                let cosDiff = abs(cosSwift - cosF64[t][d])
                let sinDiff = abs(sinSwift - sinF64[t][d])
                if cosDiff > maxCosDiff {
                    maxCosDiff = cosDiff
                    maxDiffTok = t
                    maxDiffFreq = d
                    maxDiffMid = extMids[t]
                }
                maxSinDiff = max(maxSinDiff, sinDiff)
            }
        }

        print("[RESULT] cos max-abs-diff: \(maxCosDiff)  sin max-abs-diff: \(maxSinDiff)")
        print("[RESULT] worst diff at token=\(maxDiffTok) (mid=\(maxDiffMid)) freq=\(maxDiffFreq)")

        // Negative positions span tokens 50..99 in our test grid.
        // Compute max diff restricted to the negative range.
        var maxNegCosDiff: Double = 0
        var maxNegSinDiff: Double = 0
        for t in 50..<100 {
            for d in 0..<dHalf {
                let cosSwift = Double(cosArr[t * dHalf + d])
                let sinSwift = Double(sinArr[t * dHalf + d])
                maxNegCosDiff = max(maxNegCosDiff, abs(cosSwift - cosF64[t][d]))
                maxNegSinDiff = max(maxNegSinDiff, abs(sinSwift - sinF64[t][d]))
            }
        }
        print("[RESULT-NEG] cos max-abs-diff (negative tokens): \(maxNegCosDiff)  sin: \(maxNegSinDiff)")

        // Tolerance: should be at the f32 noise floor since both compute
        // in F64 then cast to F32. ~1e-6 expected.
        #expect(maxCosDiff < 1e-5, "Swift double-precision RoPE diverges from Python on extended audio positions (max cos diff \(maxCosDiff))")
        #expect(maxSinDiff < 1e-5, "Swift double-precision RoPE diverges from Python on extended audio positions (max sin diff \(maxSinDiff))")
    }
}
