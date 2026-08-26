// DownloadByteProgress.swift - Byte-level progress for multi-file model downloads
// Copyright 2026

import Foundation

/// Session-level `URLSession` delegate routing download progress and
/// completion per task.
///
/// `URLSession.download(for:)` returns only when the whole file has landed, so
/// without progress a 26 GB component is one silent wait — indistinguishable
/// from a hang, which is what app integrators reported.
///
/// This has to be a **session** delegate driving an explicit `downloadTask`.
/// The obvious-looking `download(for:delegate:)` overload does not work:
/// measured against a real 32 MB transfer with `Content-Length` present, its
/// per-task delegate received **zero** `didWriteData` calls. That overload
/// takes delivery of the file itself and only forwards `URLSessionTaskDelegate`
/// concerns (authentication, redirects), not download progress.
///
/// `@unchecked Sendable`: all mutable state is the `handlers` dictionary, and
/// every access to it is under `lock`.
final class DownloadCoordinator: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    /// What one in-flight task needs to report and where its bytes go.
    private struct Handlers {
        let destination: URL
        let onBytes: (@Sendable (Int64, Int64) -> Void)?
        /// Carries the HTTP status rather than the `URLResponse`, which is not
        /// `Sendable` and would have to cross the continuation boundary.
        let finish: @Sendable (Result<Int?, Error>) -> Void
        /// Set by `didFinishDownloadingTo` when moving the file failed, so
        /// `didCompleteWithError` reports that rather than success.
        var moveError: Error?
    }

    private let lock = NSLock()
    private var handlers: [Int: Handlers] = [:]

    /// Register a task before resuming it.
    func register(
        taskIdentifier: Int,
        destination: URL,
        onBytes: (@Sendable (Int64, Int64) -> Void)?,
        finish: @escaping @Sendable (Result<Int?, Error>) -> Void
    ) {
        lock.lock(); defer { lock.unlock() }
        handlers[taskIdentifier] = Handlers(
            destination: destination, onBytes: onBytes, finish: finish)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        lock.lock()
        let onBytes = handlers[downloadTask.taskIdentifier]?.onBytes
        lock.unlock()
        // `totalBytesExpectedToWrite` is NSURLSessionTransferSizeUnknown (-1)
        // when the server sends no length; the receiver decides what to do.
        onBytes?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    /// The temporary file is deleted as soon as this returns, so the move must
    /// happen here rather than after the continuation resumes.
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        lock.lock()
        let destination = handlers[downloadTask.taskIdentifier]?.destination
        lock.unlock()
        guard let destination else { return }

        do {
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            // A stale file at the destination would make moveItem throw. The
            // caller already skips files that exist, so anything here is debris
            // from an interrupted run.
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            lock.lock()
            handlers[downloadTask.taskIdentifier]?.moveError = error
            lock.unlock()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        lock.lock()
        let entry = handlers.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        guard let entry else { return }

        let status = (task.response as? HTTPURLResponse)?.statusCode
        if let error {
            entry.finish(.failure(error))
        } else if let moveError = entry.moveError {
            entry.finish(.failure(moveError))
        } else {
            entry.finish(.success(status))
        }
    }
}

/// Rate-limits progress callbacks.
///
/// `didWriteData` fires per network chunk — tens of thousands of times for a
/// 42 GB file. Forwarding every one of those turns a progress bar into a
/// bottleneck, so emit only on a meaningful move.
final class ProgressThrottle: @unchecked Sendable {
    private let lock = NSLock()
    private var lastEmitted: Double = -1

    /// Minimum change in the aggregate fraction worth reporting. 0.001 is
    /// ~0.1 %: fine enough that a 26 GB file's bar never looks stuck (it moves
    /// roughly every 26 MB), coarse enough to stay cheap.
    private let step: Double

    init(step: Double = 0.001) {
        self.step = step
    }

    /// True when `fraction` is far enough from the last reported value, or is a
    /// terminal 1.0 that must not be swallowed.
    func shouldEmit(_ fraction: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard fraction >= lastEmitted + step || fraction >= 1.0 else { return false }
        lastEmitted = fraction
        return true
    }
}

/// A weighted, ordered set of files to fetch as one logical download.
///
/// Exists so a multi-file checkpoint reports **one** fraction that only ever
/// climbs. Reporting per-file fractions back to back makes the bar restart at
/// every file; reporting `fileIndex / fileCount` makes a 26 GB file and a
/// 0.36 GB file look equally long.
struct WeightedDownloadPlan {
    struct Item {
        /// What to name in the progress message.
        let label: String
        /// Path of the file inside the HuggingFace repository.
        let repoPath: String
        /// Declared size in GB — the item's share of the aggregate.
        let sizeGB: Double
        let destination: URL

        init(label: String, repoPath: String, sizeGB: Double, destination: URL) {
            self.label = label
            self.repoPath = repoPath
            self.sizeGB = sizeGB
            self.destination = destination
        }

        /// A catalogued checkpoint component, labelled by its filename.
        init(file: LTXComponentFile, destination: URL) {
            self.init(
                label: file.filename, repoPath: file.path,
                sizeGB: Double(file.sizeGB), destination: destination)
        }
    }

    /// Only the files actually missing from disk. A resumed download weights
    /// what is left rather than jumping to 90 % and sitting there.
    let items: [Item]

    /// Sum of the weights below. Never zero when `items` is non-empty — a zero
    /// total would make every fraction NaN.
    let totalGB: Double

    /// Declared sizes are approximations maintained by hand in the catalog. If
    /// they are all missing or zero, weight the files equally rather than
    /// dividing by zero.
    private let usesDeclaredSizes: Bool

    init(items: [Item]) {
        self.items = items
        let declared = items.reduce(0.0) { $0 + $1.sizeGB }
        self.usesDeclaredSizes = declared > 0
        self.totalGB = declared > 0 ? declared : Double(max(items.count, 1))
    }

    /// Weight of one item, honouring the equal-weight fallback above.
    func weight(of item: Item) -> Double {
        usesDeclaredSizes ? item.sizeGB : 1.0
    }

    /// Files whose destination does not exist yet.
    static func missing(_ candidates: [Item]) -> [Item] {
        candidates.filter { !FileManager.default.fileExists(atPath: $0.destination.path) }
    }
}
