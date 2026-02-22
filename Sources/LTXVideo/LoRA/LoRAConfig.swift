// LoRAConfig.swift - LoRA Configuration for LTX-2
// Copyright 2025

import Foundation

// MARK: - LoRA Configuration

/// Configuration for a LoRA (Low-Rank Adaptation) model
struct LoRAConfig: Codable, Sendable {
    /// Path to the LoRA weights file (.safetensors)
    let weightsPath: String

    /// Scale factor to apply to LoRA weights (default: 1.0)
    let scale: Float

    /// Whether to fuse LoRA weights into the base model for faster inference
    let fused: Bool

    /// Optional name/identifier for this LoRA
    let name: String?

    /// Optional scheduler overrides (for Turbo-style LoRAs)
    let schedulerOverrides: LoRASchedulerOverrides?

    init(
        weightsPath: String,
        scale: Float = 1.0,
        fused: Bool = true,
        name: String? = nil,
        schedulerOverrides: LoRASchedulerOverrides? = nil
    ) {
        self.weightsPath = weightsPath
        self.scale = scale
        self.fused = fused
        self.name = name
        self.schedulerOverrides = schedulerOverrides
    }
}

// MARK: - Scheduler Overrides

/// Scheduler configuration overrides for specialized LoRAs (e.g., Turbo)
public struct LoRASchedulerOverrides: Codable, Sendable {
    /// Override number of inference steps
    let numSteps: Int?

    /// Override guidance scale
    let guidanceScale: Float?

    /// Custom sigma schedule
    let sigmas: [Float]?

    init(
        numSteps: Int? = nil,
        guidanceScale: Float? = nil,
        sigmas: [Float]? = nil
    ) {
        self.numSteps = numSteps
        self.guidanceScale = guidanceScale
        self.sigmas = sigmas
    }
}

// MARK: - LoRA Layer Info

/// Information about a single LoRA layer
struct LoRALayerInfo: Sendable {
    /// Original weight key in the base model
    let originalKey: String

    /// LoRA down projection (A) weight key
    let downKey: String

    /// LoRA up projection (B) weight key
    let upKey: String

    /// Rank of the LoRA adaptation
    let rank: Int

    /// Alpha scaling factor (if present in weights)
    let alpha: Float?

    /// Computed scale: alpha / rank (or 1.0 if no alpha)
    var effectiveScale: Float {
        if let alpha = alpha {
            return alpha / Float(rank)
        }
        return 1.0
    }
}

// MARK: - LoRA Metadata

/// Metadata about a loaded LoRA model
struct LoRAInfo: Sendable {
    /// Name or identifier
    let name: String

    /// File path
    let path: String

    /// Total number of LoRA layers
    let layerCount: Int

    /// Rank used in the adaptation
    let rank: Int

    /// Target modules (e.g., ["to_q", "to_k", "to_v", "to_out"])
    let targetModules: [String]

    /// Whether the LoRA includes scheduler overrides
    let hasSchedulerOverrides: Bool

    /// File size in bytes
    let fileSizeBytes: Int64?
}

// MARK: - LoRA Target Types

/// Types of layers that can be targeted by LoRA
enum LoRATargetType: String, CaseIterable, Sendable {
    /// Query projection in attention
    case toQ = "to_q"

    /// Key projection in attention
    case toK = "to_k"

    /// Value projection in attention
    case toV = "to_v"

    /// Output projection in attention
    case toOut = "to_out"

    /// Feed-forward projection
    case ffProj = "ff.net"

    /// All attention projections
    static var attention: [LoRATargetType] {
        return [.toQ, .toK, .toV, .toOut]
    }

    /// All targets
    static var all: [LoRATargetType] {
        return allCases
    }
}

// MARK: - Multiple LoRAs

/// Configuration for loading multiple LoRAs with different scales
struct MultiLoRAConfig: Sendable {
    /// List of LoRA configurations
    let loras: [LoRAConfig]

    /// Whether LoRAs should be composed (multiplied) or added
    let compositionMode: LoRACompositionMode

    init(
        loras: [LoRAConfig],
        compositionMode: LoRACompositionMode = .add
    ) {
        self.loras = loras
        self.compositionMode = compositionMode
    }
}

/// How multiple LoRAs are combined
enum LoRACompositionMode: String, Sendable {
    /// Add LoRA deltas together
    case add

    /// Multiply LoRA weights (chain adaptations)
    case multiply
}
