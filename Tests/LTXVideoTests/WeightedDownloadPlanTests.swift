// WeightedDownloadPlanTests.swift — the aggregate download fraction's contract
// Copyright 2026
//
// App integrators reported a checkpoint download that looked hung: the 26.3 GB
// text encoder occupied one third of the bar and sat there for minutes, and the
// aggregate fell back to zero between the transformer and the components. Both
// are properties of the weighting, testable without a network.

import Foundation
import Testing
@testable import LTXVideo

@Suite("Weighted download plan")
struct WeightedDownloadPlanTests {

    static func item(_ label: String, _ gb: Double) -> WeightedDownloadPlan.Item {
        .init(label: label, repoPath: label, sizeGB: gb,
              destination: URL(fileURLWithPath: "/nonexistent/\(label)"))
    }

    /// The LTX-2.5 split checkpoint, as `downloadCheckpoint` plans it.
    static var ltx25Plan: WeightedDownloadPlan {
        WeightedDownloadPlan(items: [
            item("transformer", 42.0),
            item("text-encoder", 26.3),
            item("video-vae", 1.45),
            item("audio-vae", 0.36),
        ])
    }

    /// Replays what `runDownloadPlan` reports, without any I/O: for each item,
    /// the aggregate at a series of within-file fractions.
    static func aggregates(
        _ plan: WeightedDownloadPlan,
        samplesPerFile: Int = 5
    ) -> [Double] {
        var out: [Double] = []
        var completed = 0.0
        for item in plan.items {
            let weight = plan.weight(of: item)
            for s in 0...samplesPerFile {
                let fileFraction = Double(s) / Double(samplesPerFile)
                out.append((completed + fileFraction * weight) / plan.totalGB)
            }
            completed += weight
        }
        return out
    }

    @Test func totalIsTheSumOfDeclaredSizes() {
        #expect(abs(Self.ltx25Plan.totalGB - 70.11) < 0.001)
    }

    @Test func theAggregateNeverDecreases() {
        // The reported regression: downloadUnifiedWeights drove the fraction to
        // 1.0, then the shared-component loop restarted it at 0/3.
        let values = Self.aggregates(Self.ltx25Plan)
        for (a, b) in zip(values, values.dropFirst()) {
            #expect(b >= a, "fraction went backwards: \(a) → \(b)")
        }
    }

    @Test func theAggregateSpansZeroToOne() {
        let values = Self.aggregates(Self.ltx25Plan)
        #expect(values.first == 0.0)
        #expect(abs((values.last ?? 0) - 1.0) < 1e-9)
    }

    @Test func eachFileOccupiesItsSizeShare() {
        let plan = Self.ltx25Plan
        // The whole point of weighting: the transformer is 60% of the bar and
        // the audio VAE half a percent, rather than 25% each.
        #expect(abs(plan.weight(of: plan.items[0]) / plan.totalGB - 0.599) < 0.01)
        #expect(abs(plan.weight(of: plan.items[3]) / plan.totalGB - 0.005) < 0.01)
    }

    @Test func aFileAlreadyOnDiskIsNotPlanned() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let present = dir.appendingPathComponent("here.safetensors")
        try Data([0x00]).write(to: present)
        let candidates: [WeightedDownloadPlan.Item] = [
            .init(label: "here", repoPath: "here.safetensors", sizeGB: 42,
                  destination: present),
            .init(label: "missing", repoPath: "missing.safetensors", sizeGB: 26.3,
                  destination: dir.appendingPathComponent("missing.safetensors")),
        ]

        let plan = WeightedDownloadPlan(items: WeightedDownloadPlan.missing(candidates))
        // A resumed download weights what is left, so the bar spans the work
        // still to do rather than starting at 61% and crawling.
        #expect(plan.items.count == 1)
        #expect(plan.items.first?.label == "missing")
        #expect(abs(plan.totalGB - 26.3) < 0.001)
    }

    @Test func undeclaredSizesFallBackToEqualWeights() {
        // A catalog entry with no size must not turn every fraction into NaN.
        let plan = WeightedDownloadPlan(items: [Self.item("a", 0), Self.item("b", 0)])
        #expect(plan.totalGB == 2)
        #expect(plan.weight(of: plan.items[0]) == 1)
        let values = Self.aggregates(plan)
        #expect(values.allSatisfy { $0.isFinite })
        #expect(abs((values.last ?? 0) - 1.0) < 1e-9)
    }

    @Test func anEmptyPlanIsNotADivisionByZero() {
        let plan = WeightedDownloadPlan(items: [])
        #expect(plan.items.isEmpty)
        #expect(plan.totalGB > 0)
    }

    @Test func componentFileItemsLabelThemselvesByFilename() {
        let file = LTXComponentFile(
            kind: .textEncoder,
            path: "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            sizeGB: 26.3)
        let item = WeightedDownloadPlan.Item(
            file: file, destination: URL(fileURLWithPath: "/tmp/x"))
        #expect(item.label == "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        // The repo path must survive intact — it is what the URL is built from.
        #expect(item.repoPath == file.path)
        #expect(abs(item.sizeGB - 26.3) < 0.001)
    }
}

@Suite("Progress throttle")
struct ProgressThrottleTests {

    @Test func smallMovesAreSuppressed() {
        let throttle = ProgressThrottle(step: 0.01)
        #expect(throttle.shouldEmit(0.0))
        #expect(!throttle.shouldEmit(0.005))
        #expect(throttle.shouldEmit(0.02))
    }

    @Test func completionIsAlwaysEmitted() {
        // 1.0 must never be swallowed by the step check, or a finished download
        // leaves the bar just short of full.
        let throttle = ProgressThrottle(step: 0.5)
        #expect(throttle.shouldEmit(0.9))
        #expect(throttle.shouldEmit(1.0))
    }

    @Test func aFortyTwoGigabyteFileEmitsABoundedNumberOfUpdates() {
        // 0.1% steps: ~1000 callbacks over a whole download, not the tens of
        // thousands didWriteData fires.
        let throttle = ProgressThrottle()
        var emitted = 0
        for i in 0...100_000 where throttle.shouldEmit(Double(i) / 100_000.0) {
            emitted += 1
        }
        #expect(emitted <= 1100, "throttle let \(emitted) updates through")
        #expect(emitted >= 900)
    }
}
