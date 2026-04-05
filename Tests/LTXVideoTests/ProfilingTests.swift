//
//  ProfilingTests.swift
//  ltx-video-swift-mlx
//

import Testing
import Foundation
@testable import LTXVideo

// MARK: - ProfilingEvent Tests

@Suite("ProfilingEvent")
struct ProfilingEventTests {
    @Test func testCategoryThreadIds() {
        // Each category should map to a distinct thread lane
        let categories: [ProfilingCategory] = [
            .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation,
            .transformerLoad, .denoisingLoop, .denoisingStep, .transformerUnload,
            .upscaler,
            .vaeLoad, .vaeDecode,
            .audioLoad, .audioDenoise,
            .postProcess,
            .memoryOp,
            .evalSync,
            .custom,
        ]

        for category in categories {
            #expect(category.threadId > 0)
            #expect(!category.threadName.isEmpty)
        }
    }

    @Test func testTextEncodingCategoriesShareThread() {
        #expect(ProfilingCategory.textEncoderLoad.threadId == ProfilingCategory.textEncoding.threadId)
        #expect(ProfilingCategory.textEncoding.threadId == ProfilingCategory.textEncoderUnload.threadId)
    }

    @Test func testTransformerCategoriesShareThread() {
        #expect(ProfilingCategory.transformerLoad.threadId == ProfilingCategory.denoisingLoop.threadId)
        #expect(ProfilingCategory.denoisingLoop.threadId == ProfilingCategory.denoisingStep.threadId)
    }

    @Test func testEventInit() {
        let event = ProfilingEvent(
            name: "Test Phase",
            category: .textEncoding,
            phase: .begin,
            timestampUs: 1000
        )
        #expect(event.name == "Test Phase")
        #expect(event.category == .textEncoding)
        #expect(event.phase == .begin)
        #expect(event.timestampUs == 1000)
        #expect(event.durationUs == nil)
        #expect(event.threadId == ProfilingCategory.textEncoding.threadId)
    }

    @Test func testCompleteEventWithDuration() {
        let event = ProfilingEvent(
            name: "Step 1/8",
            category: .denoisingStep,
            phase: .complete,
            timestampUs: 5000,
            durationUs: 2000,
            stepIndex: 1,
            totalSteps: 8
        )
        #expect(event.durationUs == 2000)
        #expect(event.stepIndex == 1)
        #expect(event.totalSteps == 8)
    }

    @Test func testPhaseRawValues() {
        #expect(ProfilingPhase.begin.rawValue == "B")
        #expect(ProfilingPhase.end.rawValue == "E")
        #expect(ProfilingPhase.complete.rawValue == "X")
        #expect(ProfilingPhase.instant.rawValue == "i")
        #expect(ProfilingPhase.counter.rawValue == "C")
        #expect(ProfilingPhase.metadata.rawValue == "M")
    }
}

// MARK: - ProfilingConfig Tests

@Suite("ProfilingConfig")
struct ProfilingConfigTests {
    @Test func testDefaultConfig() {
        let config = ProfilingConfig()
        #expect(config.trackMemory == true)
        #expect(config.trackPerStepMemory == false)
        #expect(config.benchmarkRuns == nil)
        #expect(config.warmupRuns == 1)
        #expect(config.exportChromeTrace == true)
        #expect(config.printSummary == true)
    }

    @Test func testSingleRunPreset() {
        let config = ProfilingConfig.singleRun
        #expect(config.trackMemory == true)
        #expect(config.exportChromeTrace == true)
    }

    @Test func testBenchmarkPreset() {
        let config = ProfilingConfig.benchmark(runs: 5, warmup: 2)
        #expect(config.benchmarkRuns == 5)
        #expect(config.warmupRuns == 2)
        #expect(config.exportChromeTrace == false)
        #expect(config.trackMemory == true)
    }

    @Test func testDetailedPreset() {
        let config = ProfilingConfig.detailed
        #expect(config.trackMemory == true)
        #expect(config.trackPerStepMemory == true)
        #expect(config.exportChromeTrace == true)
    }
}

// MARK: - ProfilingSession Tests

@Suite("ProfilingSession")
struct ProfilingSessionTests {
    @Test func testSessionInit() {
        let session = ProfilingSession(config: .singleRun)
        #expect(!session.sessionId.isEmpty)
        #expect(session.systemRAMGB > 0)
        #expect(session.modelVariant == "")
        #expect(session.steps == 0)
    }

    @Test func testBeginEndPhase() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].phase == .begin)
        #expect(events[1].phase == .end)
        #expect(events[0].name == "Text Encoding")
        #expect(events[1].timestampUs > events[0].timestampUs)
    }

    @Test func testRecordComplete() {
        let session = ProfilingSession(config: .singleRun)
        session.recordComplete("Quick op", category: .evalSync, durationUs: 5000)

        let events = session.getEvents()
        #expect(events.count == 1)
        #expect(events[0].phase == .complete)
        #expect(events[0].durationUs == 5000)
    }

    @Test func testRecordDenoisingStep() {
        let session = ProfilingSession(config: ProfilingConfig(trackPerStepMemory: true))
        session.recordDenoisingStep(index: 1, total: 8, durationUs: 100_000)
        session.recordDenoisingStep(index: 2, total: 8, durationUs: 95_000)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].category == .denoisingStep)
        #expect(events[0].stepIndex == 1)
        #expect(events[1].stepIndex == 2)
    }

    @Test func testMemoryTimeline() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)

        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 2)
        #expect(timeline[0].context == "begin:Test")
        #expect(timeline[1].context == "end:Test")
        #expect(timeline[0].mlxActiveMB >= 0)
    }

    @Test func testMemoryTimelineDisabled() {
        let config = ProfilingConfig(trackMemory: false)
        let session = ProfilingSession(config: config)
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)

        let timeline = session.getMemoryTimeline()
        #expect(timeline.isEmpty)
    }

    @Test func testRecordMemorySnapshot() {
        let session = ProfilingSession(config: .singleRun)
        session.recordMemorySnapshot(context: "after_load")

        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 1)
        #expect(timeline[0].context == "after_load")
    }

    @Test func testGenerateReport() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "qint8"
        session.resolution = "256x256"
        session.frames = 9
        session.steps = 8

        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)

        session.beginPhase("Denoising", category: .denoisingLoop)
        session.recordDenoisingStep(index: 1, total: 2, durationUs: 50_000)
        session.recordDenoisingStep(index: 2, total: 2, durationUs: 55_000)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Denoising", category: .denoisingLoop)

        let report = session.generateReport()
        #expect(report.contains("LTX-2.3 PROFILING REPORT"))
        #expect(report.contains("distilled"))
        #expect(report.contains("qint8"))
        #expect(report.contains("Text Encoding"))
        #expect(report.contains("DENOISING STEP STATISTICS"))
        #expect(report.contains("Steps: 2"))
        #expect(report.contains("MEMORY"))
    }

    @Test func testElapsedSeconds() {
        let session = ProfilingSession(config: .singleRun)
        Thread.sleep(forTimeInterval: 0.01)
        #expect(session.elapsedSeconds > 0.005)
    }

    @Test func testMetadataFields() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "dev"
        session.quantization = "bf16"
        session.resolution = "768x512"
        session.frames = 121
        session.steps = 40

        #expect(session.modelVariant == "dev")
        #expect(session.quantization == "bf16")
        #expect(session.resolution == "768x512")
        #expect(session.frames == 121)
        #expect(session.steps == 40)
    }
}

// MARK: - Category Inference Tests

@Suite("Category Inference")
struct CategoryInferenceTests {
    @Test func testInferTextPhases() {
        #expect(ProfilingSession.inferCategory("Load Text Encoder") == .textEncoderLoad)
        #expect(ProfilingSession.inferCategory("Load Gemma") == .textEncoderLoad)
        #expect(ProfilingSession.inferCategory("Text Encoding") == .textEncoding)
        #expect(ProfilingSession.inferCategory("Text encoding phase") == .textEncoding)
        #expect(ProfilingSession.inferCategory("Unload Text Encoder") == .textEncoderUnload)
        #expect(ProfilingSession.inferCategory("Unload Gemma") == .textEncoderUnload)
    }

    @Test func testInferTransformerPhases() {
        #expect(ProfilingSession.inferCategory("Load Transformer") == .transformerLoad)
        #expect(ProfilingSession.inferCategory("Unload Transformer") == .transformerUnload)
        #expect(ProfilingSession.inferCategory("Denoising loop") == .denoisingLoop)
    }

    @Test func testInferVAEPhases() {
        #expect(ProfilingSession.inferCategory("Load VAE") == .vaeLoad)
        #expect(ProfilingSession.inferCategory("VAE Decode") == .vaeDecode)
        #expect(ProfilingSession.inferCategory("VAE decode step") == .vaeDecode)
    }

    @Test func testInferOtherPhases() {
        #expect(ProfilingSession.inferCategory("Upscaler 2x") == .upscaler)
        #expect(ProfilingSession.inferCategory("VLM prompt") == .vlmInterpretation)
        #expect(ProfilingSession.inferCategory("Prompt Enhancement") == .vlmInterpretation)
        #expect(ProfilingSession.inferCategory("Load Audio") == .audioLoad)
        #expect(ProfilingSession.inferCategory("unknown phase") == .custom)
    }
}

// MARK: - ChromeTraceExporter Tests

@Suite("ChromeTraceExporter")
struct ChromeTraceExporterTests {
    @Test func testExportSingleSession() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.beginPhase("Text Encoding", category: .textEncoding)
        session.endPhase("Text Encoding", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        #expect(data.count > 0)

        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)

        let traceEvents = json?["traceEvents"] as? [[String: Any]]
        #expect(traceEvents != nil)
        #expect((traceEvents?.count ?? 0) > 0)

        // Should contain metadata event with process name
        let metaEvents = traceEvents?.filter { ($0["ph"] as? String) == "M" } ?? []
        #expect(!metaEvents.isEmpty)

        // Should contain session info instant event
        let sessionInfo = traceEvents?.first { ($0["name"] as? String) == "Session Info" }
        #expect(sessionInfo != nil)
    }

    @Test func testExportComparison() {
        let session1 = ProfilingSession(config: .singleRun)
        session1.modelVariant = "qint8"
        session1.beginPhase("Test", category: .textEncoding)
        session1.endPhase("Test", category: .textEncoding)

        let session2 = ProfilingSession(config: .singleRun)
        session2.modelVariant = "bf16"
        session2.beginPhase("Test", category: .textEncoding)
        session2.endPhase("Test", category: .textEncoding)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "Config A", session: session1),
            (label: "Config B", session: session2),
        ])

        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let traceEvents = json?["traceEvents"] as? [[String: Any]]
        #expect(traceEvents != nil)

        // Should have events from two different processes (pid 1 and pid 2)
        let pids = Set(traceEvents?.compactMap { $0["pid"] as? Int } ?? [])
        #expect(pids.contains(1))
        #expect(pids.contains(2))
    }

    @Test func testExportIncludesMemoryCounters() {
        let session = ProfilingSession(config: .singleRun)
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let traceEvents = json?["traceEvents"] as? [[String: Any]]

        let counterEvents = traceEvents?.filter { ($0["ph"] as? String) == "C" } ?? []
        #expect(!counterEvents.isEmpty)
    }
}

// MARK: - BenchmarkRunner Tests

@Suite("BenchmarkRunner")
struct BenchmarkRunnerTests {
    @Test func testAggregateEmpty() {
        let result = BenchmarkAggregator.aggregate(sessions: [], warmupCount: 0)
        #expect(result.measuredRuns == 0)
        #expect(result.phaseStats.isEmpty)
        #expect(result.totalStats.meanMs == 0)
    }

    @Test func testAggregateSingleSession() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "qint8"
        session.resolution = "256x256"
        session.frames = 9
        session.steps = 8

        session.beginPhase("Text Encoding", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Text Encoding", category: .textEncoding)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 0)
        #expect(result.measuredRuns == 1)
        #expect(result.modelVariant == "distilled")
        #expect(!result.phaseStats.isEmpty)
        #expect(result.totalStats.meanMs > 0)
    }

    @Test func testBenchmarkReport() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "distilled"
        session.quantization = "bf16"
        session.resolution = "512x512"
        session.frames = 9
        session.steps = 8

        session.beginPhase("Denoising", category: .denoisingLoop)
        session.recordDenoisingStep(index: 1, total: 2, durationUs: 100_000)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Denoising", category: .denoisingLoop)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 1)
        let report = result.generateReport()
        #expect(report.contains("LTX-2.3 BENCHMARK REPORT"))
        #expect(report.contains("distilled"))
        #expect(report.contains("PHASE TIMINGS"))
        #expect(report.contains("MEMORY"))
    }
}

// MARK: - LTXVideoProfiler Tests

@Suite("LTXVideoProfiler", .serialized)
struct LTXVideoProfilerTests {
    @Test func testEnableDisable() {
        let profiler = LTXVideoProfiler.shared
        profiler.disable()
        profiler.reset()
        #expect(profiler.isEnabled == false)

        profiler.enable()
        #expect(profiler.isEnabled == true)
        profiler.disable()
        #expect(profiler.isEnabled == false)
    }

    @Test func testRecordTimings() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()

        profiler.start("Phase A")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Phase A")

        // Give the barrier queue time
        Thread.sleep(forTimeInterval: 0.05)

        let timings = profiler.getTimings()
        #expect(timings.count >= 1)
        #expect(timings.last?.name == "Phase A")
        #expect((timings.last?.duration ?? 0) > 0.005)

        profiler.disable()
        profiler.reset()
    }

    @Test func testRecordSteps() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()
        profiler.setTotalSteps(4)

        profiler.recordStep(duration: 1.5)
        profiler.recordStep(duration: 1.4)

        Thread.sleep(forTimeInterval: 0.05)

        let steps = profiler.getStepTimes()
        #expect(steps.count >= 2)

        profiler.disable()
        profiler.reset()
    }

    @Test func testSessionBridge() {
        let profiler = LTXVideoProfiler.shared
        let session = ProfilingSession(config: .singleRun)
        profiler.enable()
        profiler.activeSession = session

        profiler.start("Bridged Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Bridged Phase")

        Thread.sleep(forTimeInterval: 0.05)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].name == "Bridged Phase")
        #expect(events[0].phase == .begin)
        #expect(events[1].phase == .end)

        profiler.activeSession = nil
        profiler.disable()
        profiler.reset()
    }

    @Test func testDisabledProfilerNoOps() {
        let profiler = LTXVideoProfiler.shared
        profiler.disable()
        profiler.reset()

        profiler.start("Should not record")
        profiler.end("Should not record")
        profiler.recordStep(duration: 1.0)

        Thread.sleep(forTimeInterval: 0.05)

        let timings = profiler.getTimings()
        #expect(timings.isEmpty)
    }

    @Test func testGenerateReport() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()

        profiler.start("Text Encoding")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Text Encoding")

        Thread.sleep(forTimeInterval: 0.05)

        let report = profiler.generateReport()
        #expect(report.contains("Text Encoding"))
        #expect(report.contains("PHASE TIMINGS"))

        profiler.disable()
        profiler.reset()
    }

    @Test func testMeasureClosure() {
        let profiler = LTXVideoProfiler.shared
        profiler.enable()

        let result = profiler.measure("Computation") {
            return 42
        }

        #expect(result == 42)

        Thread.sleep(forTimeInterval: 0.05)
        let timings = profiler.getTimings()
        #expect(timings.contains(where: { $0.name == "Computation" }))

        profiler.disable()
        profiler.reset()
    }
}

// MARK: - TimingEntry Tests

@Suite("TimingEntry")
struct TimingEntryTests {
    @Test func testDurationFormatMs() {
        let entry = TimingEntry(
            name: "fast", duration: 0.05,
            startTime: Date(), endTime: Date()
        )
        #expect(entry.durationMs == 50.0)
        #expect(entry.durationFormatted.contains("ms"))
    }

    @Test func testDurationFormatSeconds() {
        let entry = TimingEntry(
            name: "medium", duration: 5.5,
            startTime: Date(), endTime: Date()
        )
        #expect(entry.durationFormatted.contains("s"))
        #expect(!entry.durationFormatted.contains("ms"))
        #expect(!entry.durationFormatted.contains("m"))
    }

    @Test func testDurationFormatMinutes() {
        let entry = TimingEntry(
            name: "long", duration: 125.3,
            startTime: Date(), endTime: Date()
        )
        #expect(entry.durationFormatted.contains("m"))
    }
}

// MARK: - MemoryTimelineEntry Tests

@Suite("MemoryTimelineEntry")
struct MemoryTimelineEntryTests {
    @Test func testInit() {
        let entry = MemoryTimelineEntry(
            timestampUs: 1000,
            context: "test_point",
            mlxActiveMB: 1024.5,
            mlxCacheMB: 256.0,
            mlxPeakMB: 2048.0,
            processFootprintMB: 4096.0
        )
        #expect(entry.timestampUs == 1000)
        #expect(entry.context == "test_point")
        #expect(entry.mlxActiveMB == 1024.5)
        #expect(entry.processFootprintMB == 4096.0)
    }
}
