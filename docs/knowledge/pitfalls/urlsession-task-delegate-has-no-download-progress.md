---
type: Pitfall
title: URLSession's per-task delegate never reports download progress
description: download(for:delegate:) accepts a URLSessionDownloadDelegate and calls didWriteData zero times. Byte progress needs a session-level delegate driving an explicit downloadTask.
tags: [downloads, urlsession, progress, concurrency]
timestamp: 2026-08-26T00:00:00Z
---

`URLSession.download(for:delegate:)` takes a `URLSessionTaskDelegate`, and
`URLSessionDownloadDelegate` refines it, so this compiles and looks correct:

```swift
let delegate = MyProgressDelegate(onBytes: …)   // implements didWriteData
let (url, response) = try await session.download(for: request, delegate: delegate)
```

**`didWriteData` is called zero times.** Measured against a real 32 MB transfer
from HuggingFace with `Content-Length: 32169626` present in the response:

```
status: 200
Content-Length header: 32169626
expectedContentLength: 32169626
delegate calls: 0
```

The async overload takes delivery of the file itself and only forwards
`URLSessionTaskDelegate` concerns — authentication challenges, redirects,
metrics. Download-specific progress is not among them.

It fails silently in the worst way: the download succeeds, the file is correct,
and the progress callback simply never fires. A `guard expected > 0 else
{ return }` inside the callback makes it look like a missing-`Content-Length`
problem, which sends you chasing the server instead of the API.

# What works

A **session**-level delegate plus an explicit `downloadTask`:

```swift
let coordinator = DownloadCoordinator()
session = URLSession(configuration: config, delegate: coordinator, delegateQueue: serialQueue)

let task = session.downloadTask(with: request)
let status: Int? = try await withCheckedThrowingContinuation { continuation in
    coordinator.register(taskIdentifier: task.taskIdentifier, destination: destination,
                         onBytes: onBytes, finish: { continuation.resume(with: $0) })
    task.resume()
}
```

See `Sources/LTXVideo/Utils/DownloadByteProgress.swift`.

# Two consequences of doing it this way

- **The temporary file dies when `didFinishDownloadingTo` returns.** The move to
  the final destination has to happen inside that delegate call, not after the
  continuation resumes.
- **The body lands on disk before anyone checks the status.** A 404 or 401 page
  gets moved into place looking exactly like a cached checkpoint, and every
  later run "finds" it and skips the download. Delete the destination on any
  non-200 — covered by `aFailedRequestLeavesNothingCached`.

# Also

Resume the continuation from `didCompleteWithError`, not from
`didFinishDownloadingTo`: only the former runs on the failure path, and it is
where the task's `response` (hence the status code) is available. Pass the
status code across the boundary rather than the `URLResponse`, which is not
`Sendable`.
