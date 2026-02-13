// LTXQuantizationConfig.swift - Quantization Configuration
// Copyright 2025

import Foundation

// MARK: - Transformer Quantization

/// Quantization options for the transformer
public enum TransformerQuantization: String, CaseIterable, Sendable {
    /// Full precision (float32)
    case float32 = "float32"

    /// Half precision (bfloat16)
    case bfloat16 = "bfloat16"

    /// 8-bit quantization
    case int8 = "int8"

    /// FP8 quantization (E4M3)
    case fp8 = "fp8"

    public var displayName: String {
        switch self {
        case .float32: return "Float32"
        case .bfloat16: return "BFloat16"
        case .int8: return "INT8"
        case .fp8: return "FP8"
        }
    }

    /// Memory reduction factor compared to float32
    public var memoryReduction: Float {
        switch self {
        case .float32: return 1.0
        case .bfloat16: return 0.5
        case .int8: return 0.25
        case .fp8: return 0.25
        }
    }
}

// MARK: - Text Encoder Quantization

/// Quantization options for the text encoder
public enum TextEncoderQuantization: String, CaseIterable, Sendable {
    /// Full precision
    case float32 = "float32"

    /// Half precision
    case bfloat16 = "bfloat16"

    /// 8-bit quantization
    case int8 = "int8"

    /// 4-bit quantization
    case int4 = "int4"

    public var displayName: String {
        switch self {
        case .float32: return "Float32"
        case .bfloat16: return "BFloat16"
        case .int8: return "8-bit"
        case .int4: return "4-bit"
        }
    }

    /// Memory reduction factor compared to float32
    public var memoryReduction: Float {
        switch self {
        case .float32: return 1.0
        case .bfloat16: return 0.5
        case .int8: return 0.25
        case .int4: return 0.125
        }
    }
}

// MARK: - Combined Configuration

/// Combined quantization configuration
public struct LTXQuantizationConfig: Sendable {
    /// Transformer quantization
    public var transformer: TransformerQuantization

    /// Text encoder quantization
    public var textEncoder: TextEncoderQuantization

    public init(
        transformer: TransformerQuantization = .bfloat16,
        textEncoder: TextEncoderQuantization = .bfloat16
    ) {
        self.transformer = transformer
        self.textEncoder = textEncoder
    }

    /// Estimated total memory usage in GB
    public var estimatedMemoryGB: Int {
        // Base estimates for LTX-2
        let baseTransformerGB: Float = 12.0  // ~2B parameters
        let baseTextEncoderGB: Float = 4.0   // Gemma3 encoder
        let baseVAEGB: Float = 2.0

        let transformerGB = baseTransformerGB * transformer.memoryReduction
        let textEncoderGB = baseTextEncoderGB * textEncoder.memoryReduction

        return Int(ceil(transformerGB + textEncoderGB + baseVAEGB))
    }

    // MARK: - Presets

    /// Default configuration (balanced)
    public static let balanced = LTXQuantizationConfig(
        transformer: .bfloat16,
        textEncoder: .bfloat16
    )

    /// Memory-efficient configuration
    public static let memoryEfficient = LTXQuantizationConfig(
        transformer: .fp8,
        textEncoder: .int8
    )

    /// Maximum quality configuration
    public static let maxQuality = LTXQuantizationConfig(
        transformer: .float32,
        textEncoder: .float32
    )

    /// Default configuration (same as balanced)
    public static let `default` = LTXQuantizationConfig.balanced
}

extension LTXQuantizationConfig: CustomStringConvertible {
    public var description: String {
        "LTXQuantizationConfig(transformer: \(transformer.displayName), textEncoder: \(textEncoder.displayName), ~\(estimatedMemoryGB)GB)"
    }
}
