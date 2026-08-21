// TemporalTilingTests.swift — the tiler's geometry contract
// Copyright 2026

import Foundation
import Testing
@preconcurrency import MLX
@testable import LTXVideo

@Suite("Temporal tiling")
struct TemporalTilingTests {

    @Test func shortCanvasIsOneTileWithNoLeadIn() {
        let tiles = TemporalTiling.tiles(latentFrames: 16, maxTileFrames: 32)
        #expect(tiles.count == 1)
        #expect(tiles[0] == TemporalTile(start: 0, endExclusive: 16, dropPrefix: 0))
    }

    @Test(arguments: [(31, 12, 2), (61, 16, 2), (121, 24, 3), (100, 10, 1), (17, 16, 4)])
    func ownedRangesTileTheCanvasExactly(_ frames: Int, _ maxTile: Int, _ lead: Int) {
        let tiles = TemporalTiling.tiles(
            latentFrames: frames, maxTileFrames: maxTile, leadFrames: lead)
        // No gap, no double ownership, nothing past the end.
        #expect(TemporalTiling.ownedRangesCover(tiles, latentFrames: frames),
                "frames \(frames) tile \(maxTile) lead \(lead): \(tiles)")
        for tile in tiles {
            #expect(tile.length <= maxTile, "tile \(tile) exceeds the budget")
            #expect(tile.endExclusive <= frames)
            #expect(tile.dropPrefix < tile.length)
        }
        #expect(tiles[0].dropPrefix == 0, "the first tile owns its start")
    }

    @Test func everyLaterTileOverlapsItsPredecessor() {
        let tiles = TemporalTiling.tiles(latentFrames: 61, maxTileFrames: 16, leadFrames: 2)
        #expect(tiles.count > 1)
        for (previous, tile) in zip(tiles, tiles.dropFirst()) {
            // The lead-in must fall inside what the previous tile already denoised,
            // otherwise the seam is a hard cut rather than a shared boundary.
            #expect(tile.start < previous.endExclusive)
            #expect(tile.dropPrefix > 0)
            #expect(tile.start + tile.dropPrefix == previous.endExclusive)
        }
    }

    @Test func leadInIsClampedNotAssumed() {
        // A lead longer than what precedes it must clamp rather than index
        // before the start of the canvas.
        let tiles = TemporalTiling.tiles(latentFrames: 20, maxTileFrames: 6, leadFrames: 10)
        #expect(TemporalTiling.ownedRangesCover(tiles, latentFrames: 20))
        for tile in tiles { #expect(tile.start >= 0) }
    }
}

@Suite("Densified clip rate")
struct DensifiedClipRateTests {

    @Test func aDensifiedClipKeepsItsDuration() {
        // A temporal round doubles the frames and the rate, so the last frame of
        // the refined clip must sit at the same second as the last frame of the
        // source. Positioning the dense clip at the source's rate instead reads
        // as a clip of twice the duration: half-speed motion to the model, and on
        // a long canvas, coordinates past the 20 s RoPE range.
        let sourceLatentFrames = 16          // 121 pixel frames
        let denseLatentFrames = 2 * sourceLatentFrames - 1

        let sourceEnd = LTXPipeline.gridTemporalPosition(
            latentFrame: sourceLatentFrames - 1, fps: 24.0)
        let denseEnd = LTXPipeline.gridTemporalPosition(
            latentFrame: denseLatentFrames - 1, fps: 48.0)
        // Within one dense latent frame's span (8 pixel frames at 48 fps): the
        // causal shift makes the two grids' last midpoints differ slightly, but
        // the clips must end at the same moment, not at twice it.
        #expect(abs(denseEnd - sourceEnd) < 8.0 / 48.0, "\(denseEnd) s vs \(sourceEnd) s")

        // The same clip positioned at the source rate lands at twice the time.
        let wrong = LTXPipeline.gridTemporalPosition(
            latentFrame: denseLatentFrames - 1, fps: 24.0)
        #expect(wrong > 1.9 * sourceEnd)
    }

    @Test func anchorsAndTheGridShareOneRate() {
        // The anchor coordinate must land exactly on the base grid's coordinate
        // for the same latent frame — a half-frame offset is enough to weaken it.
        for frame in [0, 1, 7, 30] {
            let grid = createPositionGrid(
                batchSize: 1, frames: 31, height: 1, width: 1, fps: 48.0)
            let expected = grid[0, 0, frame].item(Float.self)
            let anchor = LTXPipeline.gridTemporalPosition(latentFrame: frame, fps: 48.0)
            #expect(abs(anchor - expected) < 1e-6, "frame \(frame): \(anchor) vs \(expected)")
        }
    }
}

@Suite("Generated keyframes as tile anchors")
struct SlotAnchorPlacementTests {

    /// A single window covering a 121-frame source densified to 241.
    static let whole = TemporalTile(start: 0, endExclusive: 31, dropPrefix: 0)

    @Test func aSlotLandsAtTwiceItsSourceFrame() {
        // The round doubles the frame count at constant duration, so the moment
        // a slot captured at source frame 40 is at dense frame 80. Anchoring it
        // at 40 would pin it a whole second early.
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 0, tile: Self.whole) == 0)
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 40, tile: Self.whole) == 80)
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 80, tile: Self.whole) == 160)
    }

    @Test func aTileRebasesToItsOwnOrigin() {
        // Tiles are denoised standalone with positions restarting at 0. A tile
        // beginning at latent frame 16 covers pixels from 8·16 − 7 = 121, so a
        // slot at source frame 80 (dense 160) sits at local 39.
        let tile = TemporalTile(start: 16, endExclusive: 31, dropPrefix: 2)
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 80, tile: tile) == 160 - 121)
        // Everything before the window is refused rather than clamped to 0 —
        // clamping would stack every earlier anchor onto the tile's first frame.
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 40, tile: tile) == nil)
    }

    @Test func slotsOutsideTheWindowAreRefused() {
        // The last pixel a 31-frame window covers is (31 − 1)·8 = 240.
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 120, tile: Self.whole) == 240)
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 121, tile: Self.whole) == nil)
    }

    @Test func theFirstLatentFrameIsClampedLikeTheGrid() {
        // The causal grid collapses frame 0's span, so a tile starting at 0 has
        // pixel origin 0, not −7. An unclamped origin would shift every anchor
        // in the first tile by seven frames.
        let first = TemporalTile(start: 0, endExclusive: 8, dropPrefix: 0)
        #expect(LTXPipeline.slotLocalPixel(sourceFrame: 1, tile: first) == 2)
    }
}
