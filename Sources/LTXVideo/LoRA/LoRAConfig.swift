// LoRAConfig.swift - LoRA Configuration for LTX-2
// Copyright 2025

import Foundation

// MARK: - LoRA Configuration

/// Configuration for a LoRA (Low-Rank Adaptation) model
public struct LoRAConfig: Codable, Sendable {
    /// Path to the LoRA weights file (.safetensors)
    public let weightsPath: String

    /// Scale factor to apply to LoRA weights (default: 1.0)
    public let scale: Float

    /// Whether to fuse LoRA weights into the base model for faster inference
    public let fused: Bool

    /// Optional name/identifier for this LoRA
    public let name: String?

    /// Optional scheduler overrides (for Turbo-style LoRAs)
    public let schedulerOverrides: LoRASchedulerOverrides?

    public init(
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
    public let numSteps: Int?

    /// Override guidance scale
    public let guidanceScale: Float?

    /// Custom sigma schedule
    public let sigmas: [Float]?

    public init(
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
public struct LoRALayerInfo: Sendable {
    /// Original weight key in the base model
    public let originalKey: String

    /// LoRA down projection (A) weight key
    public let downKey: String

    /// LoRA up projection (B) weight key
    public let upKey: String

    /// Rank of the LoRA adaptation
    public let rank: Int

    /// Alpha scaling factor (if present in weights)
    public let alpha: Float?

    /// Computed scale: alpha / rank (or 1.0 if no alpha)
    public var effectiveScale: Float {
        if let alpha = alpha {
            return alpha / Float(rank)
        }
        return 1.0
    }
}

// MARK: - LoRA Metadata

/// Metadata about a loaded LoRA model
public struct LoRAInfo: Sendable {
    /// Name or identifier
    public let name: String

    /// File path
    public let path: String

    /// Total number of LoRA layers
    public let layerCount: Int

    /// Rank used in the adaptation
    public let rank: Int

    /// Target modules (e.g., ["to_q", "to_k", "to_v", "to_out"])
    public let targetModules: [String]

    /// Whether the LoRA includes scheduler overrides
    public let hasSchedulerOverrides: Bool

    /// File size in bytes
    public let fileSizeBytes: Int64?
}

// MARK: - LoRA Target Types

/// Types of layers that can be targeted by LoRA
public enum LoRATargetType: String, CaseIterable, Sendable {
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
    public static var attention: [LoRATargetType] {
        return [.toQ, .toK, .toV, .toOut]
    }

    /// All targets
    public static var all: [LoRATargetType] {
        return allCases
    }
}

// MARK: - Multiple LoRAs

/// Configuration for loading multiple LoRAs with different scales
public struct MultiLoRAConfig: Sendable {
    /// List of LoRA configurations
    public let loras: [LoRAConfig]

    /// Whether LoRAs should be composed (multiplied) or added
    public let compositionMode: LoRACompositionMode

    public init(
        loras: [LoRAConfig],
        compositionMode: LoRACompositionMode = .add
    ) {
        self.loras = loras
        self.compositionMode = compositionMode
    }
}

/// How multiple LoRAs are combined
public enum LoRACompositionMode: String, Sendable {
    /// Add LoRA deltas together
    case add

    /// Multiply LoRA weights (chain adaptations)
    case multiply
}
