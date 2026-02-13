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
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
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
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // MARK: - CLI
        .executableTarget(
            name: "LTXVideoCLI",
            dependencies: [
                "LTXVideo",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // MARK: - Tests
        .testTarget(
            name: "LTXVideoTests",
            dependencies: ["LTXVideo"]
        ),
    ]
)
