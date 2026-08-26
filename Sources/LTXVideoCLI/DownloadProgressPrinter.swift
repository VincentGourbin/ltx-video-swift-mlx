// DownloadProgressPrinter.swift - One-line rendering of download progress
// Copyright 2026

import Foundation
import LTXVideo

/// Renders ``DownloadProgress`` for a terminal.
///
/// Byte-level progress fires roughly a thousand times per download. Printing a
/// line each time — which is what every call site did when the only events were
/// one per file — buries everything else in scrollback. On a TTY this rewrites
/// a single line; piped to a file, where carriage returns are noise, it falls
/// back to a line every 10 %.
final class DownloadProgressPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastLineLength = 0
    private var lastMilestone = -1
    private let isTerminal: Bool

    init(isTerminal: Bool = isatty(fileno(stdout)) == 1) {
        self.isTerminal = isTerminal
    }

    /// A callback to hand to `ModelDownloader`.
    func callback() -> DownloadProgressCallback {
        { [self] progress in render(progress) }
    }

    /// Report one event. Equivalent to invoking ``callback()``, for call sites
    /// that already have a closure of their own.
    func report(_ progress: DownloadProgress) {
        render(progress)
    }

    /// Close the current line, if one is open. Call before printing anything
    /// else, or the next line lands on top of the progress line.
    func finish() {
        lock.lock(); defer { lock.unlock() }
        guard isTerminal, lastLineLength > 0 else { return }
        print()
        lastLineLength = 0
    }

    private func render(_ progress: DownloadProgress) {
        let percent = Int((progress.progress * 100).rounded())
        let line = Self.describe(progress, percent: percent)

        lock.lock(); defer { lock.unlock() }

        guard isTerminal else {
            // Only on crossing a 10 % boundary, so a captured log stays short.
            let milestone = percent / 10
            guard milestone > lastMilestone else { return }
            lastMilestone = percent >= 100 ? -1 : milestone
            print("  \(line)")
            return
        }

        // Pad to erase whatever the previous, possibly longer, line left behind.
        let padding = max(0, lastLineLength - line.count)
        print("\r  \(line)\(String(repeating: " ", count: padding))", terminator: "")

        // Close the line as soon as a phase completes, so the caller's next
        // print does not land on top of it. Saves every call site from having
        // to remember `finish()`.
        if percent >= 100 {
            print()
            lastLineLength = 0
            lastMilestone = -1
        } else {
            lastLineLength = line.count
        }
        fflush(stdout)
    }

    private static func describe(_ progress: DownloadProgress, percent: Int) -> String {
        guard progress.totalBytes > 0, progress.bytesDownloaded > 0 else {
            return "\(progress.message) (\(percent)%)"
        }
        let done = Self.formatBytes(progress.bytesDownloaded)
        let total = Self.formatBytes(progress.totalBytes)
        let file = progress.currentFile ?? progress.message
        return "\(file)  \(done) / \(total)  (\(percent)%)"
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        if gb >= 1 { return String(format: "%.2f GB", gb) }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576.0)
    }
}

/// Shared renderer. One CLI invocation runs one command, so a single instance
/// keeps the "erase the previous line" state coherent across the phases of a
/// run (checkpoint, then auxiliary models, then the enhancer).
let downloadPrinter = DownloadProgressPrinter()
