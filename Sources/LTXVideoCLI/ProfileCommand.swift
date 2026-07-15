// ProfileCommand.swift - CLI profiling and benchmarking commands
// Copyright 2026 Vincent Gourbin

import Foundation
import ArgumentParser
import LTXVideo

// MARK: - Profile Command Group

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile and benchmark LTX-2.3 inference pipeline",
        subcommands: [ProfileRun.self, ProfileBenchmark.self, ProfileCompare.self],
        defaultSubcommand: ProfileRun.self
    )
}

// MARK: - Shared Options

struct ProfileModelOptions: ParsableArguments {
    @Argument(help: "Text prompt for video generation")
    var prompt: String

    @Option(name: .shortAndLong, help: "Video width (must be divisible by 64)")
    var width: Int = 768

    @Option(name: .shortAndLong, help: "Video height (must be divisible by 64)")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1)")
    var frames: Int = 25

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "bf16"

    @Option(name: .long, help: "Input image for I2V generation")
    var image: String?

    @Flag(name: .long, help: "Skip saving the generated video")
    var noVideo: Bool = false

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory")
    var modelsDir: String?

    @Option(name: .long, help: "Path to local Gemma model directory")
    var gemmaPath: String?

    @Option(name: .long, help: "Path to local LTX unified weights file")
    var ltxWeights: String?

    func validate() throws {
        guard width % 64 == 0 && height % 64 == 0 else {
            throw ValidationError("Width and height must be divisible by 64. Got \(width)x\(height)")
        }
        guard (frames - 1) % 8 == 0 else {
            throw ValidationError("Frame count must be 8n+1 (e.g., 9, 17, 25). Got \(frames)")
        }
        guard TransformerQuantization(rawValue: transformerQuant) != nil else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use: bf16, qint8, or int4")
        }
    }

    func resolveQuantization() -> LTXQuantizationConfig {
        LTXQuantizationConfig(transformer: TransformerQuantization(rawValue: transformerQuant)!)
    }
}

// MARK: - Profile Run

struct ProfileRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Single profiled generation with Chrome Trace export"
    )

    @OptionGroup var options: ProfileModelOptions

    @Option(name: .long, help: "Output directory for trace files")
    var outputDir: String = "./profile_results"

    @Flag(name: .long, help: "Record memory at each denoising step")
    var perStepMemory: Bool = false

    @Flag(name: .long, help: "Disable Chrome Trace JSON export")
    var noChromeTrace: Bool = false

    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

    func run() async throws {
        if let dir = options.modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        RuntimeBeacon.isEnabled = beacon

        let quantConfig = options.resolveQuantization()
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        let config = ProfilingConfig(
            trackMemory: true, trackPerStepMemory: perStepMemory,
            outputDirectory: URL(fileURLWithPath: outputDir),
            exportChromeTrace: !noChromeTrace, printSummary: true
        )

        let session = ProfilingSession(config: config)
        session.metadata["model"] = "distilled"
        session.metadata["quant"] = options.transformerQuant
        session.metadata["resolution"] = "\(options.width)x\(options.height)"
        session.metadata["frames"] = String(options.frames)
        session.metadata["steps"] = String(8)

        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.activeSession = session
        defer {
            profiler.activeSession = nil
            profiler.disable()
        }

        print("Profiling: LTX-2.3 distilled \(options.transformerQuant)")
        print("Resolution: \(options.width)x\(options.height)  Frames: \(options.frames)  Steps: 8")
        print()

        let pipeline = LTXPipeline(model: .distilled, quantization: quantConfig, hfToken: options.hfToken)
        try await pipeline.loadModels(gemmaModelPath: options.gemmaPath, ltxWeightsPath: options.ltxWeights)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()

        let genConfig = LTXVideoGenerationConfig(
            width: options.width, height: options.height, numFrames: options.frames,
            numSteps: 8, seed: options.seed, imagePath: options.image
        )

        let result = try await pipeline.generateVideo(
            prompt: options.prompt, config: genConfig, upscalerWeightsPath: upscalerPath,
            onProgress: { progress in print("\r  \(progress.status)", terminator: ""); fflush(stdout) },
        )
        print()

        if !options.noVideo {
            let videoPath = "\(outputDir)/profiled_output.mp4"
            let _ = try await VideoExporter.exportVideo(
                frames: result.frames, width: options.width, height: options.height,
                fps: 24.0, to: URL(fileURLWithPath: videoPath)
            )
            print("Video saved to \(videoPath)")
        }

        print(session.generateReport())

        if !noChromeTrace {
            let traceData = ChromeTraceExporter.export(session: session)
            let tracePath = "\(outputDir)/ltx23_\(options.transformerQuant)_trace.json"
            try traceData.write(to: URL(fileURLWithPath: tracePath))
            print("Chrome Trace exported to \(tracePath)")
            print("View in Perfetto: https://ui.perfetto.dev/")
        }
    }
}

// MARK: - Profile Benchmark

struct ProfileBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Statistical benchmarking with warm-up and multiple runs"
    )

    @OptionGroup var options: ProfileModelOptions

    @Option(name: .long, help: "Number of warm-up runs (excluded from stats)")
    var warmup: Int = 1

    @Option(name: .long, help: "Number of measured runs")
    var runs: Int = 3

    @Option(name: .long, help: "Output directory for results")
    var outputDir: String = "./benchmark_results"

    func run() async throws {
        if let dir = options.modelsDir {
            LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir)
        }

        let quantConfig = options.resolveQuantization()
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        print("Benchmarking: LTX-2.3 distilled \(options.transformerQuant)")
        print("Resolution: \(options.width)x\(options.height)  Frames: \(options.frames)  Steps: 8")
        print("Warm-up: \(warmup), Measured runs: \(runs)")
        print()

        let pipeline = LTXPipeline(model: .distilled, quantization: quantConfig, hfToken: options.hfToken)
        try await pipeline.loadModels(gemmaModelPath: options.gemmaPath, ltxWeightsPath: options.ltxWeights)
        let upscalerPath = try await pipeline.downloadUpscalerWeights()
        let profiler = LTXVideoProfiler.shared

        let totalRuns = warmup + runs
        var measuredSessions: [ProfilingSession] = []

        for i in 0..<totalRuns {
            let isWarmup = i < warmup
            let runLabel = isWarmup ? "Warm-up \(i + 1)/\(warmup)" : "Run \(i - warmup + 1)/\(runs)"
            print("\(runLabel)...", terminator: " "); fflush(stdout)

            let session = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            session.metadata["model"] = "distilled"
            session.metadata["quant"] = options.transformerQuant
            session.metadata["resolution"] = "\(options.width)x\(options.height)"
            session.metadata["frames"] = String(options.frames)
            session.metadata["steps"] = String(8)

            profiler.enable()
            profiler.activeSession = session
            defer { profiler.activeSession = nil; profiler.disable() }

            let genConfig = LTXVideoGenerationConfig(
                width: options.width, height: options.height, numFrames: options.frames,
                numSteps: 8, seed: options.seed ?? 42, imagePath: options.image
            )

            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = try await pipeline.generateVideo(
                prompt: options.prompt, config: genConfig, upscalerWeightsPath: upscalerPath
            )
            print(String(format: "%.2fs", CFAbsoluteTimeGetCurrent() - startTime))

            if !isWarmup { measuredSessions.append(session) }
        }

        print()
        let result = BenchmarkAggregator.aggregate(sessions: measuredSessions, warmupCount: warmup)
        print(result.generateReport())

        let labeledSessions = measuredSessions.enumerated().map { (i, s) in (label: "Run \(i + 1)", session: s) }
        let traceData = ChromeTraceExporter.exportComparison(sessions: labeledSessions)
        let tracePath = "\(outputDir)/benchmark_ltx23_\(runs)runs.json"
        try traceData.write(to: URL(fileURLWithPath: tracePath))
        print("Benchmark trace exported to \(tracePath)")
    }
}

// MARK: - Profile Compare

struct ProfileCompare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compare",
        abstract: "Compare performance across quantization configurations"
    )

    @Argument(help: "Text prompt for video generation")
    var prompt: String

    @Option(name: .long, help: "Quantization configs to compare (comma-separated, e.g. 'bf16,qint8,int4')")
    var configs: String

    @Option(name: .shortAndLong, help: "Video width") var width: Int = 512
    @Option(name: .shortAndLong, help: "Video height") var height: Int = 512
    @Option(name: .shortAndLong, help: "Number of frames (must be 8n+1)") var frames: Int = 9
    @Option(name: .long, help: "Random seed") var seed: UInt64 = 42
    @Option(name: .long, help: "HuggingFace token") var hfToken: String?
    @Option(name: .long, help: "Custom models directory") var modelsDir: String?
    @Option(name: .long, help: "Path to local Gemma model directory") var gemmaPath: String?
    @Option(name: .long, help: "Path to local LTX unified weights file") var ltxWeights: String?
    @Option(name: .long, help: "Output directory") var outputDir: String = "./comparison_results"

    func run() async throws {
        if let dir = modelsDir { LTXModelRegistry.customModelsDirectory = URL(fileURLWithPath: dir) }

        let quantNames = configs.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard !quantNames.isEmpty else {
            throw ValidationError("No configurations specified. Use --configs 'bf16,qint8,int4'")
        }
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        print("Comparing \(quantNames.count) quantization configs")
        print("Resolution: \(width)x\(height)  Frames: \(frames)  Seed: \(seed)\n")

        var labeledSessions: [(label: String, session: ProfilingSession)] = []
        let profiler = LTXVideoProfiler.shared

        for quantName in quantNames {
            guard let quantOption = TransformerQuantization(rawValue: quantName) else {
                print("Skipping invalid quant: \(quantName)"); continue
            }

            let label = "distilled \(quantName)"
            print("Running: \(label)...")

            let session = ProfilingSession(config: ProfilingConfig(trackMemory: true))
            session.metadata["model"] = "distilled"
            session.metadata["quant"] = quantName
            session.metadata["resolution"] = "\(width)x\(height)"
            session.metadata["frames"] = String(frames)
            session.metadata["steps"] = String(8)

            profiler.enable()
            profiler.activeSession = session
            defer { profiler.activeSession = nil; profiler.disable() }

            let pipeline = LTXPipeline(
                model: .distilled, quantization: LTXQuantizationConfig(transformer: quantOption), hfToken: hfToken
            )
            try await pipeline.loadModels(gemmaModelPath: gemmaPath, ltxWeightsPath: ltxWeights)
            let upscalerPath = try await pipeline.downloadUpscalerWeights()

            let genConfig = LTXVideoGenerationConfig(
                width: width, height: height, numFrames: frames, numSteps: 8, seed: seed
            )

            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = try await pipeline.generateVideo(
                prompt: prompt, config: genConfig, upscalerWeightsPath: upscalerPath
            )
            print("  \(label): \(String(format: "%.2fs", CFAbsoluteTimeGetCurrent() - startTime))")
            labeledSessions.append((label: label, session: session))
        }

        print("\nCOMPARISON SUMMARY")
        print(String(repeating: "-", count: 60))
        for (label, session) in labeledSessions {
            let events = session.getEvents()
            var totalMs: Double = 0
            var begins: [String: UInt64] = [:]
            for event in events {
                if event.phase == .begin { begins[event.name] = event.timestampUs }
                else if event.phase == .end, let b = begins[event.name] {
                    totalMs += Double(event.timestampUs - b) / 1000.0; begins.removeValue(forKey: event.name)
                }
            }
            let peakMLX = session.getMemoryTimeline().map(\.mlxActiveMB).max() ?? 0
            print("  \(label.padding(toLength: 25, withPad: " ", startingAt: 0)) \(String(format: "%8.2fs", totalMs / 1000))  Peak: \(String(format: "%.0f", peakMLX))MB")
        }
        print()

        let traceData = ChromeTraceExporter.exportComparison(sessions: labeledSessions)
        let tracePath = "\(outputDir)/comparison_trace.json"
        try traceData.write(to: URL(fileURLWithPath: tracePath))
        print("Comparison trace exported to \(tracePath)")
        print("View in Perfetto: https://ui.perfetto.dev/")
    }
}
