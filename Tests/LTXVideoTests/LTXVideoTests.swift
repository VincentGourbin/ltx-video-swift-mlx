//
//  LTXVideoTests.swift
//  ltx-video-swift-mlx
//

import Testing
@testable import LTXVideo

@Test func testVersion() async throws {
    #expect(LTXVideo.version == "0.1.0")
}
