// InterpolateCommand.swift - Double a clip's frame rate
// Copyright 2026

import ArgumentParser
import Foundation
import LTXVideo

/// Doubles the frame rate of an existing clip: `n` frames become `2n - 1` at
/// the same duration, so motion reads smoother.
///
/// The latent temporal upsampler lays the motion onto a denser grid, then a
/// short ancestral refinement invents the in-between motion — without it the
/// result is an interpolation and looks like one.
struct Interpolate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Double a video's frame rate with LTX-2.5's temporal upsampler"
    )

    @Argument(help: "Prompt describing the clip — the refinement still conditions on text")
    var prompt: String

    @Option(name: .shortAndLong, help: "Source video")
    var input: String

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "interpolated.mp4"

    @Option(name: .shortAndLong, help: "Source width (must match the clip)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Source height (must match the clip)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Source frame count (8n+1)")
    var frames: Int = 121

    @Option(name: .long, help: "Output frame rate; defaults to twice the source's 24 fps")
    var fps: Int = 48

    @Option(name: .long, help: "How ancestral the refinement is: 0 interpolates, 1 invents most")
    var eta: Float = 0.5

    @Option(name: .long, help: "Noise level the refinement starts from; defaults to 0.975 for a single window and 0.725 when the clip is tiled, where independent tiles at a high level stop agreeing at their seams")
    var renoiseFrom: Float?

    @Option(name: .long, help: "Anchor every Nth source frame to hold the subject (0 disables); defaults to 4 for a single window, 1 when tiled")
    var anchorEvery: Int?

    @Option(name: .long, help: "Max latent frames denoised at once; lower trades speed for memory on long clips")
    var tileFrames: Int = 32

    @Option(name: .long, help: "Source frame rate; the refined clip is positioned at twice it, capped at 60")
    var sourceFps: Float = 24.0

    @Flag(name: .long, help: "Anchor each tile on the previous tile's output as well as on the source. Measured slightly worse than the tiled defaults for twice the time; kept for experimentation")
    var carryForward: Bool = false

    @Option(name: .long, help: "Random seed")
    var seed: UInt64?

    @Option(name: .long, help: "Model variant (temporal upsampling ships with LTX-2.5)")
    var model: String = "2.5-distilled"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "bf16"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom directory for model storage")
    var modelsDir: String?

    @Flag(name: .long, help: "Enable debug output")
    var debug: Bool = false

    mutating func run() async throws {
        if let dir = modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }
        if debug { LTXDebug.enableDebugMode() }

        let variant = try parseModelVariant(model)
        guard variant.family == .ltx25 else {
            throw ValidationError(
                "Temporal upsampling ships from LTX-2.5 onward; \(variant.displayName) has none.")
        }
        guard let quantization = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid quantization: \(transformerQuant)")
        }
        guard FileManager.default.fileExists(atPath: input) else {
            throw ValidationError("Source video not found: \(input)")
        }
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1; got \(frames)")
        }

        print("\(variant.displayName) — Temporal interpolation")
        print("=======================================")
        print("Source: \(input) — \(frames) frames")
        print("Output: \(frames * 2 - 1) frames at \(fps) fps (same duration)")
        print()

        let pipeline = LTXPipeline(
            model: variant,
            quantization: LTXQuantizationConfig(transformer: quantization, textEncoder: quantization),
            hfToken: hfToken)

        print("Loading models (this may take a while)...")
        try await pipeline.loadModels { progress in
            print("  \(progress.message) (\(Int(progress.progress * 100))%)")
            fflush(stdout)
        }

        print("Fetching the temporal upsampler...")
        let downloader = ModelDownloader(hfToken: hfToken)
        let upscalerPath = try await downloader.downloadAuxiliaryModel(.latentTemporalUpscalerX2_25) {
            progress in print("  \(progress.message)"); fflush(stdout)
        }.path

        let start = Date()
        let result = try await pipeline.interpolateTemporally(
            videoPath: input, prompt: prompt, upscalerPath: upscalerPath,
            width: width, height: height, numFrames: frames, seed: seed, eta: eta,
            renoiseFrom: renoiseFrom, anchorEvery: anchorEvery, maxTileLatentFrames: tileFrames,
            sourceFPS: sourceFps, carryForward: carryForward,
            onProgress: { progress in
                print("  Step \(progress.currentStep + 1)/\(progress.totalSteps) [\(progress.phase)]")
                fflush(stdout)
            })
        print("Interpolated in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")

        _ = try await VideoExporter.exportVideo(
            frames: result.frames, width: width, height: height, fps: Double(fps),
            to: URL(fileURLWithPath: output))
        print("Saved to: \(output)")
    }
}
