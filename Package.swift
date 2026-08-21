// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ltx-video-swift-mlx",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        // Libraries
        .library(
            name: "LTXVideo",
            targets: ["LTXVideo"]),
        // CLI Tool
        .executable(
            name: "ltx-video",
            targets: ["LTXVideoCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),
        // LTX-2.5 encodes prompts with `gemma4-12b-ltx-v1`, a gemma4_unified derivative.
        // Reused rather than re-ported: that package already implements the architecture
        // (global_head_dim, attention_k_eq_v, layer_scalar, partial-rotary RoPE).
        // 1.5.0 carries the n-gram window that can skip the thinking channel,
        // without which reasoning and loop protection could not both be on.
        .package(
            url: "https://github.com/VincentGourbin/gemma-4-swift-mlx",
            from: "1.5.0"
        ),
    ],
    targets: [
        // MARK: - Library
        .target(
            name: "LTXVideo",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
                .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
            ]
        ),
        // MARK: - CLI
        .executableTarget(
            name: "LTXVideoCLI",
            dependencies: [
                "LTXVideo",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        // MARK: - Tests
        .testTarget(
            name: "LTXVideoTests",
            dependencies: ["LTXVideo"]
        ),
    ]
)
