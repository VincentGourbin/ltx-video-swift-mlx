// SlotDecodeProbeTests.swift — decode generated keyframe slots to images
// Copyright 2026
//
// A slot is only useful if it holds the *scene at its own frame*. That cannot be
// checked from a latent's statistics, so this probe decodes the slots a run
// wrote with `--slots-out` and saves them as PNGs, which are then comparable
// against the same frames of the clip the run produced.
//
//   TEST_RUNNER_LTX_SLOTS=/path/slots.safetensors \
//   TEST_RUNNER_LTX_VAE=/path/ltx-2.5-video-vae-conv-bf16.safetensors \
//   TEST_RUNNER_LTX_SLOTS_OUT=/path/outdir \
//   xcodebuild ... test -only-testing:LTXVideoTests/SlotDecodeProbeTests

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Generated slot decode probe",
       .enabled(if: ProcessInfo.processInfo.environment["LTX_SLOTS"] != nil))
struct SlotDecodeProbeTests {

    static func writePNG(_ frame: MLXArray, to url: URL) throws {
        // frame: (H, W, 3) in [0, 1]
        let h = frame.dim(0), w = frame.dim(1)
        let clamped = MLX.clip(frame * 255, min: MLXArray(Float(0)), max: MLXArray(Float(255)))
        MLX.eval(clamped)
        let values = clamped.asType(.float32).asArray(Float.self)
        var bytes = [UInt8](repeating: 255, count: h * w * 4)
        for i in 0 ..< (h * w) {
            bytes[i * 4 + 0] = UInt8(values[i * 3 + 0])
            bytes[i * 4 + 1] = UInt8(values[i * 3 + 1])
            bytes[i * 4 + 2] = UInt8(values[i * 3 + 2])
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw LTXError.invalidConfiguration("cannot write \(url.path)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        #expect(CGImageDestinationFinalize(dest))
    }

    /// Load a PNG as (1, 3, 1, H, W) in [-1, 1] — the encoder's input layout.
    static func loadPNG(_ url: URL) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw LTXError.fileNotFound(url.path)
        }
        let h = image.height, w = image.width
        var bytes = [UInt8](repeating: 0, count: h * w * 4)
        let ctx = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [Float](repeating: 0, count: h * w * 3)
        for i in 0 ..< (h * w) {
            rgb[i * 3 + 0] = Float(bytes[i * 4 + 0]) / 255
            rgb[i * 3 + 1] = Float(bytes[i * 4 + 1]) / 255
            rgb[i * 3 + 2] = Float(bytes[i * 4 + 2]) / 255
        }
        let hwc = MLXArray(rgb, [h, w, 3])
        return (hwc.transposed(2, 0, 1).expandedDimensions(axes: [0, 2]) * 2 - 1)
    }

    /// Control for the slot decode: a *real* frame put through the same
    /// single-latent-frame round trip. Softness that shows up here is the VAE's,
    /// not the slot's — the decoder normally sees latent frames spanning eight
    /// pixel frames, and a slot deliberately spans one.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["LTX_SLOT_CONTROL_FRAME"] != nil))
    func singleFrameRoundTripIsTheControl() throws {
        let env = ProcessInfo.processInfo.environment
        let source = try Self.loadPNG(URL(fileURLWithPath: env["LTX_SLOT_CONTROL_FRAME"]!))
        MLX.eval(source)

        let encoder = VideoEncoder()
        try LTXWeightLoader.applyVAEEncoderWeights(
            try LTXWeightLoader.loadVAEEncoderWeights(from: env["LTX_VAE"]!), to: encoder)
        let decoder = VideoDecoder()
        try LTXWeightLoader.applyVAEWeights(
            try LTXWeightLoader.loadVAEWeights(from: env["LTX_VAE"]!), to: decoder)
        eval(encoder.parameters(), decoder.parameters())

        let latent = encoder(source)
        MLX.eval(latent)
        #expect(latent.dim(2) == 1, "a one-frame clip is one latent frame")

        let decoded = decodeVideo(latent: latent, decoder: decoder, timestep: nil)
        MLX.eval(decoded)
        let frame = decoded[0, 0..., 0..., 0...]
        let outDir = URL(fileURLWithPath: env["LTX_SLOTS_OUT"] ?? NSTemporaryDirectory())
        try Self.writePNG(frame, to: outDir.appendingPathComponent("control-roundtrip.png"))

        let original = ((source[0, 0..., 0, 0..., 0...].transposed(1, 2, 0)) + 1) / 2
        let mse = MLX.mean(MLX.square(frame - original)).item(Float.self)
        print(String(format: "PROBE control round trip: PSNR %.2f dB, spread %.3f",
                     10 * log10(1 / max(mse, 1e-9)),
                     frame.max().item(Float.self) - frame.min().item(Float.self)))
    }

    @Test func decodesSlotsToImages() throws {
        let env = ProcessInfo.processInfo.environment
        let (tensors, _) = try MLX.loadArraysAndMetadata(
            url: URL(fileURLWithPath: env["LTX_SLOTS"]!))
        let keyframes = try #require(tensors["keyframes"]).asType(.float32)
        let indices = try #require(tensors["pixel_frame_indices"]).asArray(Int32.self)

        // (1, C, K, H, W), one latent frame per slot, at the target's spatial grid.
        #expect(keyframes.dim(0) == 1)
        #expect(keyframes.dim(2) == indices.count)
        let finite = MLX.abs(keyframes).max().item(Float.self)
        #expect(finite.isFinite && finite < 1e3, "slot latents are not finite: max |x| \(finite)")

        let weights = try LTXWeightLoader.loadVAEWeights(from: env["LTX_VAE"]!)
        let decoder = VideoDecoder()
        try LTXWeightLoader.applyVAEWeights(weights, to: decoder)
        eval(decoder.parameters())

        let mean = decoder.meanOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])
        let std = decoder.stdOfMeans.asType(.float32).reshaped([1, -1, 1, 1, 1])

        let outDir = URL(fileURLWithPath: env["LTX_SLOTS_OUT"] ?? NSTemporaryDirectory())
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        // One slot at a time: each is a standalone single-frame latent, and
        // decoding them together would let the decoder's temporal kernels mix
        // frames that are seconds apart in the clip.
        for (slot, pixelFrame) in indices.enumerated() {
            let one = keyframes[0..., 0..., slot ..< (slot + 1), 0..., 0...] * std + mean
            let decoded = decodeVideo(latent: one, decoder: decoder, timestep: nil)
            MLX.eval(decoded)
            #expect(decoded.dim(0) == 1, "one latent frame decodes to one pixel frame")
            let frame = decoded[0, 0..., 0..., 0...]
            let spread = frame.max().item(Float.self) - frame.min().item(Float.self)
            #expect(spread > 0.05, "slot \(pixelFrame) decoded to a flat image")
            try Self.writePNG(frame, to: outDir.appendingPathComponent("slot-\(pixelFrame).png"))
            print("PROBE slot \(pixelFrame): \(decoded.dim(1))x\(decoded.dim(2)), spread \(spread)")
        }
    }
}
