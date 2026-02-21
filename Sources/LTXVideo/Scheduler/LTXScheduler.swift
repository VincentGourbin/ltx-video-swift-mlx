// LTXScheduler.swift - Flow-Matching Scheduler for LTX-2
// Copyright 2025

import Foundation
import MLX

// MARK: - Constants

/// Base shift anchor for sigma calculation
private let BASE_SHIFT_ANCHOR = 1024

/// Maximum shift anchor for sigma calculation
private let MAX_SHIFT_ANCHOR = 4096

// MARK: - Distilled Sigma Values

/// Official distilled sigma schedule from LTX-2 (9 values for 8 steps)
public let DISTILLED_SIGMA_VALUES: [Float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

/// Stage 2 distilled sigma values (for two-stage generation)
public let STAGE_2_DISTILLED_SIGMA_VALUES: [Float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

// MARK: - LTX2 Scheduler

/// Flow-matching Euler scheduler for LTX-2 diffusion sampling
///
/// Generates a sigma schedule with token-count-dependent shifting.
/// Supports both standard and distilled modes.
public class LTXScheduler: @unchecked Sendable {
    /// Number of training timesteps
    public let numTrainTimesteps: Int

    /// Current sigmas (noise levels in [0, 1] range)
    public private(set) var sigmas: [Float] = []

    /// Current step index during sampling
    public private(set) var stepIndex: Int = 0

    /// Whether using distilled mode
    public private(set) var isDistilled: Bool = false

    public init(numTrainTimesteps: Int = 1000, isDistilled: Bool = false) {
        self.numTrainTimesteps = numTrainTimesteps
        self.isDistilled = isDistilled
    }

    // MARK: - Timestep Setting

    /// Set timesteps for inference
    ///
    /// - Parameters:
    ///   - numSteps: Number of denoising steps
    ///   - distilled: Whether to use distilled sigma values
    ///   - latentTokenCount: Optional token count for shift calculation
    ///   - maxShift: Maximum shift value for large token counts
    ///   - baseShift: Base shift value for small token counts
    ///   - stretch: Whether to stretch sigmas to terminal value
    ///   - terminal: Target terminal sigma value
    public func setTimesteps(
        numSteps: Int,
        distilled: Bool = false,
        latentTokenCount: Int? = nil,
        maxShift: Float = 2.05,
        baseShift: Float = 0.95,
        stretch: Bool = true,
        terminal: Float = 0.1
    ) {
        self.isDistilled = distilled
        self.stepIndex = 0

        if distilled {
            // Use predefined distilled sigma values
            self.sigmas = DISTILLED_SIGMA_VALUES
            LTXDebug.log("Scheduler set: distilled mode with \(sigmas.count - 1) steps")
        } else {
            // Compute sigma schedule
            // Clamp token count to MAX_SHIFT_ANCHOR (matching Python: min(num_tokens, MAX_SHIFT_ANCHOR))
            let tokenCount = min(latentTokenCount ?? MAX_SHIFT_ANCHOR, MAX_SHIFT_ANCHOR)

            // Linear spacing from 1.0 to 0.0
            var sigmaValues: [Float] = []
            for i in 0...numSteps {
                sigmaValues.append(1.0 - Float(i) / Float(numSteps))
            }

            // Compute shift based on token count (linear interpolation)
            let x1 = Float(BASE_SHIFT_ANCHOR)
            let x2 = Float(MAX_SHIFT_ANCHOR)
            let mm = (maxShift - baseShift) / (x2 - x1)
            let b = baseShift - mm * x1
            let sigmaShift = Float(tokenCount) * mm + b

            // Apply sigmoid-like transformation
            let expShift = exp(sigmaShift)

            sigmaValues = sigmaValues.map { sigma in
                if sigma == 0 {
                    return 0
                }
                return expShift / (expShift + pow(1.0 / sigma - 1.0, 1.0))
            }

            // Stretch sigmas so that final non-zero value matches terminal
            if stretch && numSteps > 0 {
                let oneMinusSigmas = sigmaValues.map { 1.0 - $0 }

                // Get the last non-zero sigma's (1 - sigma) value
                let lastOneMinus = oneMinusSigmas[numSteps - 1]

                // Scale factor to stretch to terminal
                let scaleFactor = lastOneMinus / (1.0 - terminal)

                // Apply stretching
                sigmaValues = sigmaValues.enumerated().map { i, sigma in
                    if sigma == 0 {
                        return 0
                    }
                    let stretched = 1.0 - (oneMinusSigmas[i] / scaleFactor)
                    return stretched
                }
            }

            self.sigmas = sigmaValues
            LTXDebug.log("Scheduler set: \(numSteps) steps, shift=\(sigmaShift), tokens=\(tokenCount)")
        }

        LTXDebug.verbose("Sigmas: \(sigmas.prefix(5))... to \(sigmas.suffix(2))")
    }

    /// Set custom sigma values directly
    ///
    /// - Parameter customSigmas: Pre-computed sigma schedule
    public func setCustomSigmas(_ customSigmas: [Float]) {
        guard !customSigmas.isEmpty else {
            LTXDebug.log("Warning: Empty custom sigmas provided")
            return
        }

        // Ensure terminal sigma 0.0 is present
        var sigmasWithTerminal = customSigmas
        if let last = customSigmas.last, last != 0.0 {
            sigmasWithTerminal.append(0.0)
        }

        self.sigmas = sigmasWithTerminal
        self.stepIndex = 0
        self.isDistilled = false

        LTXDebug.log("Custom sigmas set: \(sigmas.count - 1) effective steps")
    }

    // MARK: - Sampling

    /// Perform one Euler step
    ///
    /// - Parameters:
    ///   - modelOutput: Predicted velocity from transformer
    ///   - sample: Current noisy sample
    /// - Returns: Updated sample for next step
    public func step(modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        guard stepIndex < sigmas.count - 1 else {
            return sample
        }

        let sigma = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]

        let result = step(
            latent: sample,
            velocity: modelOutput,
            sigma: sigma,
            sigmaNext: sigmaNext
        )

        stepIndex += 1

        return result
    }

    /// Get the current sigma for this step
    public var currentSigma: Float {
        guard stepIndex < sigmas.count else { return 0 }
        return sigmas[stepIndex]
    }

    /// Get the initial sigma (for noise initialization)
    public var initialSigma: Float {
        sigmas.first ?? 1.0
    }

    /// Current progress (0.0 to 1.0)
    public var progress: Float {
        guard sigmas.count > 1 else { return 0 }
        return Float(stepIndex) / Float(sigmas.count - 1)
    }

    /// Remaining steps
    public var remainingSteps: Int {
        max(0, sigmas.count - 1 - stepIndex)
    }

    /// Total effective steps
    public var totalSteps: Int {
        max(0, sigmas.count - 1)
    }

    /// Reset scheduler state
    public func reset() {
        stepIndex = 0
    }

    /// Get sigma schedule for the given number of steps
    ///
    /// - Parameter numSteps: Number of denoising steps
    /// - Returns: Array of sigmas including terminal 0.0
    public func getSigmas(numSteps: Int) -> [Float] {
        setTimesteps(numSteps: numSteps, distilled: isDistilled)
        return sigmas
    }

    /// Perform one Euler step with explicit sigma values
    ///
    /// Matches Python mlx-video (venv_ltx2) behavior exactly:
    ///
    /// 1. to_denoised: compute in float32, cast back to original dtype
    ///    ```python
    ///    noisy_f32 = noisy.astype(mx.float32)
    ///    velocity_f32 = velocity.astype(mx.float32)
    ///    sigma_f32 = sigma.astype(mx.float32)
    ///    result = noisy_f32 - sigma_f32 * velocity_f32
    ///    return result.astype(original_dtype)
    ///    ```
    ///
    /// 2. Euler step: compute in float32, cast back to original dtype
    ///    ```python
    ///    latents_f32 = latents.astype(mx.float32)
    ///    denoised_f32 = denoised.astype(mx.float32)
    ///    latents = (denoised_f32 + sigma_next_f32 * (latents_f32 - denoised_f32) / sigma_f32).astype(dtype)
    ///    ```
    ///
    /// The round-trip (bfloat16 → float32 → bfloat16) in to_denoised followed by
    /// (bfloat16 → float32) in Euler step means denoised loses precision at the
    /// bfloat16 boundary. This matches Python's behavior exactly.
    ///
    /// - Parameters:
    ///   - latent: Current latent sample
    ///   - velocity: Predicted velocity from transformer
    ///   - sigma: Current sigma
    ///   - sigmaNext: Next sigma
    /// - Returns: Updated latent
    public func step(
        latent: MLXArray,
        velocity: MLXArray,
        sigma: Float,
        sigmaNext: Float
    ) -> MLXArray {
        let dtype = latent.dtype

        // to_denoised: compute in float32, cast back to original dtype (matching Python)
        let sigmaF32 = MLXArray(sigma).asType(.float32)
        let denoised = (latent.asType(.float32) - sigmaF32 * velocity.asType(.float32)).asType(dtype)

        if sigmaNext > 0 {
            // Euler step: compute in float32, cast back to original dtype (matching Python)
            let latentsF32 = latent.asType(.float32)
            let denoisedF32 = denoised.asType(.float32)
            let sigmaNF32 = MLXArray(sigmaNext).asType(.float32)
            return (denoisedF32 + sigmaNF32 * (latentsF32 - denoisedF32) / sigmaF32).asType(dtype)
        } else {
            // Last step: use denoised prediction directly
            return denoised
        }
    }

    // MARK: - Noise Operations

    /// Add noise to latents for a given sigma
    ///
    /// For flow matching: x_t = (1 - t) * x_0 + t * noise
    public func addNoise(
        originalSamples: MLXArray,
        noise: MLXArray,
        sigma: Float
    ) -> MLXArray {
        let t = MLXArray(sigma)
        return (1 - t) * originalSamples + t * noise
    }

    /// Scale noise for the current sigma
    public func scaleNoise(
        sample: MLXArray,
        sigma: Float,
        noise: MLXArray
    ) -> MLXArray {
        let t = MLXArray(sigma)
        return (1 - t) * sample + t * noise
    }

    /// Get velocity target for training
    public func getVelocity(sample: MLXArray, noise: MLXArray) -> MLXArray {
        // Velocity target: v = noise - sample
        return noise - sample
    }
}

// MARK: - Convenience Functions

/// Get sigma schedule for diffusion sampling
///
/// - Parameters:
///   - numSteps: Number of denoising steps
///   - distilled: If true, use predefined distilled sigma values
///   - latentTokenCount: Optional token count for shift calculation
/// - Returns: Sigma schedule as array
public func getSigmaSchedule(
    numSteps: Int,
    distilled: Bool = false,
    latentTokenCount: Int? = nil
) -> [Float] {
    if distilled {
        return DISTILLED_SIGMA_VALUES
    }

    let scheduler = LTXScheduler()
    scheduler.setTimesteps(
        numSteps: numSteps,
        distilled: false,
        latentTokenCount: latentTokenCount
    )
    return scheduler.sigmas
}
