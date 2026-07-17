// TrainingState.swift - Persistent training state for pause/resume
// Copyright 2026

import Foundation

/// Persistent training state saved as JSON alongside checkpoints.
///
/// Enables resuming training after a pause, quit, or crash.
/// Saved automatically at each checkpoint.
///
/// ## Checkpoint Directory Structure
/// ```
/// output_dir/
///   checkpoint-step500.safetensors
///   checkpoint-step500.json        ← TrainingState
///   checkpoint-step1000.safetensors
///   checkpoint-step1000.json
///   lora-final.safetensors
///   learning_curve.svg
///   .pause                         ← sentinel file
/// ```
public struct TrainingState: Codable, Sendable {

    // MARK: - Progress

    /// Current training step (1-indexed, last completed step)
    public var currentStep: Int

    /// Total training steps configured
    public var totalSteps: Int

    // MARK: - Loss Tracking

    /// Recent loss values (last 100)
    public var recentLosses: [Float]

    /// Best loss observed
    public var bestLoss: Float

    /// Step at which best loss was observed
    public var bestLossStep: Int

    /// Smoothed average loss (last 20 values)
    public var averageLoss: Float {
        guard !recentLosses.isEmpty else { return 0 }
        let window = recentLosses.suffix(20)
        return window.reduce(0, +) / Float(window.count)
    }

    // MARK: - Timing

    /// When training started
    public var startedAt: Date

    /// When this state was last saved
    public var lastCheckpointAt: Date

    /// Total training time in seconds (accumulated across pause/resume cycles)
    public var totalTrainingTime: TimeInterval

    /// Estimated time remaining in seconds
    public var estimatedTimeRemaining: TimeInterval {
        guard currentStep > 0 else { return 0 }
        let timePerStep = totalTrainingTime / Double(currentStep)
        return timePerStep * Double(totalSteps - currentStep)
    }

    // MARK: - Config Verification

    /// Hash of key config values to detect incompatible resume attempts
    public var configHash: String

    /// Model variant used
    public var modelType: String

    /// LoRA rank
    public var loraRank: Int

    /// LoRA alpha
    public var loraAlpha: Float

    /// Learning rate
    public var learningRate: Float

    /// LR schedule the run was started with ("cosine"/"constant"). Optional so
    /// pre-July-2026 checkpoints still decode; nil means the checkpoint predates
    /// the schedule and was trained under the then-only behavior, "constant".
    public var lrSchedule: String?

    /// RNG seed (if set)
    public var rngSeed: UInt64?

    // MARK: - Status

    /// Training status at time of save
    public var status: TrainingStatus

    /// Whether this was a pause checkpoint (temporary, deleted on resume)
    public var isPauseCheckpoint: Bool

    // MARK: - Init

    public init(
        totalSteps: Int,
        modelType: String,
        loraRank: Int,
        loraAlpha: Float,
        learningRate: Float,
        lrSchedule: String? = nil,
        rngSeed: UInt64? = nil
    ) {
        self.currentStep = 0
        self.totalSteps = totalSteps
        self.recentLosses = []
        self.bestLoss = .infinity
        self.bestLossStep = 0
        self.startedAt = Date()
        self.lastCheckpointAt = Date()
        self.totalTrainingTime = 0
        self.modelType = modelType
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.learningRate = learningRate
        self.lrSchedule = lrSchedule
        self.rngSeed = rngSeed
        self.status = .idle
        self.isPauseCheckpoint = false

        self.configHash = Self.computeHash(
            modelType: modelType, rank: loraRank, alpha: loraAlpha, lr: learningRate
        )
    }

    // MARK: - Loss Recording

    public mutating func recordLoss(_ loss: Float, atStep step: Int) {
        recentLosses.append(loss)
        if recentLosses.count > 100 {
            recentLosses.removeFirst(recentLosses.count - 100)
        }
        if loss < bestLoss {
            bestLoss = loss
            bestLossStep = step
        }
        currentStep = step
    }

    // MARK: - Persistence

    /// Save state to a JSON file.
    public func save(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: URL(fileURLWithPath: path))
    }

    /// Load state from a JSON file.
    public static func load(from path: String) throws -> TrainingState {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(TrainingState.self, from: data)
    }

    /// Find the latest checkpoint state in an output directory.
    ///
    /// Scans for `checkpoint-stepN.json` files and returns the highest step.
    /// - Returns: Tuple of (step, path) or nil if no checkpoints found.
    public static func findLatestCheckpoint(in outputDir: String) -> (step: Int, path: String)? {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(atPath: outputDir) else { return nil }

        var latest: (step: Int, path: String)?
        for filename in contents {
            guard filename.hasPrefix("checkpoint-step"),
                  filename.hasSuffix(".json") else { continue }
            let stepStr = filename
                .replacingOccurrences(of: "checkpoint-step", with: "")
                .replacingOccurrences(of: ".json", with: "")
            guard let step = Int(stepStr) else { continue }
            if latest == nil || step > latest!.step {
                let path = (outputDir as NSString).appendingPathComponent(filename)
                latest = (step, path)
            }
        }

        return latest
    }

    /// Verify that a checkpoint is compatible with the current config.
    ///
    /// The LR schedule must match too: resuming a constant-LR run under cosine
    /// (or vice versa) would silently switch the LR trajectory mid-run. A
    /// checkpoint without a recorded schedule (pre-July-2026) was trained under
    /// the then-only behavior, "constant".
    public func isCompatible(modelType: String, rank: Int, alpha: Float, lr: Float,
                             lrSchedule schedule: String) -> Bool {
        let otherHash = Self.computeHash(modelType: modelType, rank: rank, alpha: alpha, lr: lr)
        return configHash == otherHash && (lrSchedule ?? "constant") == schedule
    }

    // MARK: - Private

    private static func computeHash(modelType: String, rank: Int, alpha: Float, lr: Float) -> String {
        let input = "\(modelType)|\(rank)|\(alpha)|\(lr)"
        // djb2 hash
        var hash: UInt64 = 5381
        for char in input.utf8 {
            hash = ((hash &<< 5) &+ hash) &+ UInt64(char)
        }
        return String(hash, radix: 16)
    }
}
