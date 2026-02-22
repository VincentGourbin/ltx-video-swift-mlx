// LTXQuantizationConfig.swift - Quantization Configuration
// Copyright 2025

import Foundation

// MARK: - Transformer Quantization

/// Quantization options for the LTX-2 transformer.
///
/// On-the-fly quantization replaces all `Linear` layers with `QuantizedLinear`
/// after weight loading, reducing memory usage at the cost of some quality.
///
/// ## Memory estimates (LTX-2 transformer only)
/// | Option | Bits | Transformer RAM |
/// |--------|------|-----------------|
/// | bf16   | 16   | ~25 GB          |
/// | qint8  | 8    | ~13 GB          |
/// | int4   | 4    | ~7 GB           |
public enum TransformerQuantization: String, CaseIterable, Sendable {
    /// BFloat16 — full precision (default)
    case bf16 = "bf16"

    /// 8-bit quantization (qint8)
    case qint8 = "qint8"

    /// 4-bit quantization (int4)
    case int4 = "int4"

    public var displayName: String {
        switch self {
        case .bf16: return "BFloat16"
        case .qint8: return "8-bit (qint8)"
        case .int4: return "4-bit (int4)"
        }
    }

    /// Number of bits per weight
    public var bits: Int {
        switch self {
        case .bf16: return 16
        case .qint8: return 8
        case .int4: return 4
        }
    }

    /// Group size for quantization (standard MLX value)
    public var groupSize: Int { 64 }

    /// Whether on-the-fly quantization is needed
    public var needsQuantization: Bool {
        self != .bf16
    }

    /// Approximate memory reduction factor compared to bf16
    public var memoryReduction: Float {
        switch self {
        case .bf16: return 1.0
        case .qint8: return 0.5
        case .int4: return 0.25
        }
    }
}

// MARK: - Combined Configuration

/// Quantization configuration for the LTX-2 pipeline.
///
/// Controls on-the-fly quantization of the transformer model.
/// The text encoder (Gemma 3 12B) is always loaded with its pre-quantized
/// weights (4-bit from mlx-community) — no separate quantization option needed.
///
/// ## Usage
/// ```swift
/// // Default: full precision
/// let pipeline = LTXPipeline(model: .distilled)
///
/// // Memory-efficient: 8-bit transformer
/// let pipeline = LTXPipeline(
///     model: .distilled,
///     quantization: LTXQuantizationConfig(transformer: .qint8)
/// )
///
/// // Ultra-minimal: 4-bit transformer
/// let pipeline = LTXPipeline(
///     model: .distilled,
///     quantization: .minimal
/// )
/// ```
public struct LTXQuantizationConfig: Sendable {
    /// Transformer quantization level
    public var transformer: TransformerQuantization

    public init(
        transformer: TransformerQuantization = .bf16
    ) {
        self.transformer = transformer
    }

    // MARK: - Presets

    /// Default: full precision bf16 (no quantization)
    public static let `default` = LTXQuantizationConfig(transformer: .bf16)

    /// Memory-efficient: 8-bit transformer (~50% memory reduction)
    public static let memoryEfficient = LTXQuantizationConfig(transformer: .qint8)

    /// Minimal memory: 4-bit transformer (~75% memory reduction)
    public static let minimal = LTXQuantizationConfig(transformer: .int4)
}

extension LTXQuantizationConfig: CustomStringConvertible {
    public var description: String {
        "LTXQuantizationConfig(transformer: \(transformer.displayName))"
    }
}
