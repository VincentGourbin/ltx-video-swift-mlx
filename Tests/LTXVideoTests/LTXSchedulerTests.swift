//
//  LTXSchedulerTests.swift
//  ltx-video-swift-mlx
//
//  Tests for the flow-matching Euler scheduler.
//  NOTE: step() tests require Metal (MLXArray operations).

import Testing
import MLX
@testable import LTXVideo

// MARK: - Sigma Schedule Tests (no Metal needed)

@Suite("LTXScheduler Sigma Schedules")
struct LTXSchedulerSigmaTests {
    @Test func testDistilledSigmaValues() {
        #expect(DISTILLED_SIGMA_VALUES.count == 9)
        #expect(DISTILLED_SIGMA_VALUES.first == 1.0)
        #expect(DISTILLED_SIGMA_VALUES.last == 0.0)
        // Verify monotonically decreasing
        for i in 1..<DISTILLED_SIGMA_VALUES.count {
            #expect(DISTILLED_SIGMA_VALUES[i] < DISTILLED_SIGMA_VALUES[i - 1])
        }
    }

    @Test func testStage2SigmaValues() {
        #expect(STAGE_2_DISTILLED_SIGMA_VALUES.count == 4)
        #expect(STAGE_2_DISTILLED_SIGMA_VALUES.first == 0.909375)
        #expect(STAGE_2_DISTILLED_SIGMA_VALUES.last == 0.0)
        // Verify monotonically decreasing
        for i in 1..<STAGE_2_DISTILLED_SIGMA_VALUES.count {
            #expect(STAGE_2_DISTILLED_SIGMA_VALUES[i] < STAGE_2_DISTILLED_SIGMA_VALUES[i - 1])
        }
    }

    @Test func testDistilledSchedule() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        #expect(scheduler.sigmas == DISTILLED_SIGMA_VALUES)
        #expect(scheduler.totalSteps == 8)
        #expect(scheduler.isDistilled)
    }

    @Test func testNonDistilledSchedule() {
        let scheduler = LTXScheduler(isDistilled: false)
        scheduler.setTimesteps(numSteps: 30, distilled: false, latentTokenCount: 1024)
        #expect(scheduler.sigmas.count == 31) // 30 steps + terminal 0.0
        #expect(scheduler.sigmas.first! > 0.9) // starts near 1.0
        #expect(scheduler.sigmas.last == 0.0)
        // Verify monotonically decreasing
        for i in 1..<scheduler.sigmas.count {
            #expect(scheduler.sigmas[i] <= scheduler.sigmas[i - 1])
        }
    }

    @Test func testCustomSigmas() {
        let scheduler = LTXScheduler()
        scheduler.setCustomSigmas([0.9, 0.5, 0.2, 0.0])
        #expect(scheduler.sigmas == [0.9, 0.5, 0.2, 0.0])
        #expect(scheduler.totalSteps == 3)
    }

    @Test func testCustomSigmasAppendsTerminal() {
        let scheduler = LTXScheduler()
        scheduler.setCustomSigmas([0.9, 0.5, 0.2])
        #expect(scheduler.sigmas.last == 0.0)
        #expect(scheduler.sigmas.count == 4)
    }

    @Test func testCustomSigmasNoDoubleTerminal() {
        let scheduler = LTXScheduler()
        scheduler.setCustomSigmas([0.9, 0.5, 0.0])
        #expect(scheduler.sigmas == [0.9, 0.5, 0.0])
    }

    @Test func testEmptyCustomSigmas() {
        let scheduler = LTXScheduler()
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        let before = scheduler.sigmas
        scheduler.setCustomSigmas([])
        #expect(scheduler.sigmas == before)  // unchanged
    }

    @Test func testGetSigmas() {
        let scheduler = LTXScheduler(isDistilled: true)
        let sigmas = scheduler.getSigmas(numSteps: 8)
        #expect(sigmas == DISTILLED_SIGMA_VALUES)
    }

    // MARK: - Truncated Sigmas (Retake)

    @Test func testTruncatedSigmasStrength08() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        let truncated = scheduler.truncatedSigmas(forStrength: 0.8)
        // Rescaled: all 8 sigmas * 0.8, same step count
        #expect(truncated.first == 0.8)
        #expect(truncated.last == 0.0)
        #expect(truncated.count == 9) // 8 steps + terminal 0.0
    }

    @Test func testTruncatedSigmasStrength1() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        let truncated = scheduler.truncatedSigmas(forStrength: 1.0)
        // strength=1.0 → identity (same as original schedule)
        #expect(truncated.first == 1.0)
        #expect(truncated.last == 0.0)
        #expect(truncated.count == 9)
    }

    @Test func testTruncatedSigmasStrength05() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        let truncated = scheduler.truncatedSigmas(forStrength: 0.5)
        // Rescaled: all sigmas * 0.5
        #expect(truncated.first == 0.5)
        #expect(truncated.last == 0.0)
        #expect(truncated.count == 9) // same step count
    }

    @Test func testTruncatedSigmasStrengthVeryLow() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        let truncated = scheduler.truncatedSigmas(forStrength: 0.1)
        #expect(truncated.first! < 0.101) // 1.0 * 0.1 = 0.1
        #expect(truncated.last == 0.0)
        #expect(truncated.count == 9) // same step count
    }

    @Test func testTruncatedSigmasEmptySchedule() {
        let scheduler = LTXScheduler()
        let truncated = scheduler.truncatedSigmas(forStrength: 0.5)
        #expect(truncated == [0.5, 0.0])
    }

    // MARK: - Dev Model Retake (30 steps)

    @Test func testDevModelSchedule30Steps() {
        let scheduler = LTXScheduler(isDistilled: false)
        scheduler.setTimesteps(numSteps: 30, distilled: false)
        let sigmas = scheduler.sigmas
        #expect(sigmas.count == 31)  // 30 steps + terminal 0.0
        #expect(sigmas.first == 1.0)
        #expect(sigmas.last == 0.0)
        // Dev schedule should be monotonically decreasing
        for i in 1..<sigmas.count {
            #expect(sigmas[i] <= sigmas[i - 1])
        }
    }

    @Test func testDistilledRetakeFullSchedule() {
        // Retake uses full schedule (no truncation) — 8 steps
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        #expect(scheduler.sigmas.count == 9)  // 8 steps + terminal
        #expect(scheduler.sigmas.first == 1.0)
        #expect(scheduler.sigmas.last == 0.0)
    }

    // MARK: - Scheduler State

    @Test func testInitialState() {
        let scheduler = LTXScheduler(isDistilled: true)
        #expect(scheduler.stepIndex == 0)
        #expect(scheduler.sigmas.isEmpty)
        #expect(scheduler.totalSteps == 0)
        #expect(scheduler.remainingSteps == 0)
    }

    @Test func testStateAfterSetTimesteps() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        #expect(scheduler.stepIndex == 0)
        #expect(scheduler.totalSteps == 8)
        #expect(scheduler.remainingSteps == 8)
        #expect(scheduler.initialSigma == 1.0)
        #expect(scheduler.currentSigma == 1.0)
        #expect(scheduler.progress == 0.0)
    }

    @Test func testReset() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        // Manually advance step index
        _ = scheduler.step(
            modelOutput: MLXArray.zeros([1, 128]),
            sample: MLXArray.zeros([1, 128])
        )
        #expect(scheduler.stepIndex == 1)
        scheduler.reset()
        #expect(scheduler.stepIndex == 0)
    }

    @Test func testProgressAdvancement() {
        let scheduler = LTXScheduler(isDistilled: true)
        scheduler.setTimesteps(numSteps: 8, distilled: true)
        #expect(scheduler.progress == 0.0)

        // After calling step once
        _ = scheduler.step(
            modelOutput: MLXArray.zeros([1, 128]),
            sample: MLXArray.zeros([1, 128])
        )
        #expect(scheduler.stepIndex == 1)
        #expect(scheduler.remainingSteps == 7)
    }
}

// MARK: - Euler Step Tests (require Metal)

@Suite("LTXScheduler Euler Step")
struct LTXSchedulerStepTests {
    @Test func testEulerStepBasic() {
        let scheduler = LTXScheduler()
        let latent = MLXArray.ones([1, 128])
        let velocity = MLXArray.ones([1, 128]) * 0.1

        // Step from sigma=1.0 to sigma=0.5
        let result = scheduler.step(latent: latent, velocity: velocity, sigma: 1.0, sigmaNext: 0.5)
        eval(result)
        #expect(result.shape == [1, 128])
        // Denoised = latent - sigma * velocity = 1.0 - 1.0 * 0.1 = 0.9
        // Euler: denoised + sigmaNext * (latent - denoised) / sigma
        //      = 0.9 + 0.5 * (1.0 - 0.9) / 1.0 = 0.9 + 0.05 = 0.95
        let val = result.mean().item(Float.self)
        #expect(abs(val - 0.95) < 0.01)
    }

    @Test func testEulerStepFinal() {
        let scheduler = LTXScheduler()
        let latent = MLXArray.ones([1, 128])
        let velocity = MLXArray.ones([1, 128]) * 0.1

        // Final step to sigma=0.0 → returns denoised directly
        let result = scheduler.step(latent: latent, velocity: velocity, sigma: 1.0, sigmaNext: 0.0)
        eval(result)
        // Denoised = 1.0 - 1.0 * 0.1 = 0.9
        let val = result.mean().item(Float.self)
        #expect(abs(val - 0.9) < 0.01)
    }

    @Test func testEulerStepPreservesDtype() {
        let scheduler = LTXScheduler()
        let latent = MLXArray.ones([1, 128]).asType(.bfloat16)
        let velocity = MLXArray.ones([1, 128]).asType(.float32)

        let result = scheduler.step(latent: latent, velocity: velocity, sigma: 1.0, sigmaNext: 0.5)
        eval(result)
        #expect(result.dtype == .bfloat16)
    }

    @Test func testAddNoise() {
        let scheduler = LTXScheduler()
        let sample = MLXArray.ones([1, 128])
        let noise = MLXArray.zeros([1, 128])

        // sigma=0 → pure sample
        let r0 = scheduler.addNoise(originalSamples: sample, noise: noise, sigma: 0.0)
        eval(r0)
        #expect(abs(r0.mean().item(Float.self) - 1.0) < 0.01)

        // sigma=1 → pure noise
        let r1 = scheduler.addNoise(originalSamples: sample, noise: noise, sigma: 1.0)
        eval(r1)
        #expect(abs(r1.mean().item(Float.self)) < 0.01)

        // sigma=0.5 → 50/50 mix
        let r05 = scheduler.addNoise(originalSamples: sample, noise: noise, sigma: 0.5)
        eval(r05)
        #expect(abs(r05.mean().item(Float.self) - 0.5) < 0.01)
    }

    @Test func testGetVelocity() {
        let scheduler = LTXScheduler()
        let sample = MLXArray.ones([1, 128]) * 2.0
        let noise = MLXArray.ones([1, 128]) * 5.0

        let velocity = scheduler.getVelocity(sample: sample, noise: noise)
        eval(velocity)
        // v = noise - sample = 5 - 2 = 3
        #expect(abs(velocity.mean().item(Float.self) - 3.0) < 0.01)
    }

    @Test func testScaleNoise() {
        let scheduler = LTXScheduler()
        let sample = MLXArray.ones([1, 128]) * 2.0
        let noise = MLXArray.ones([1, 128]) * 3.0

        let result = scheduler.scaleNoise(sample: sample, sigma: 0.3, noise: noise)
        eval(result)
        // (1 - 0.3) * 2.0 + 0.3 * 3.0 = 1.4 + 0.9 = 2.3
        #expect(abs(result.mean().item(Float.self) - 2.3) < 0.01)
    }
}
