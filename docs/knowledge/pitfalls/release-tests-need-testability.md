---
type: Pitfall
title: Release-mode xcodebuild tests need ENABLE_TESTABILITY=YES and a dedicated derived-data path
description: Without both, the test target fails with "unable to resolve Swift module dependency to a compatible module" — and env vars need the TEST_RUNNER_ prefix to reach the test process.
tags: [xcodebuild, testing, release, e2e]
timestamp: 2026-07-16T00:00:00Z
---

Running GPU-heavy E2E tests (e.g. `LipDubReuseE2ETests`) in Debug is painfully
slow, but the naive Release invocation fails in three distinct ways:

1. **`@testable import` is Debug-only by default.** Release build of the test
   target dies with `unable to resolve Swift module dependency to a compatible
   module: 'LTXVideo'`. Fix: pass `ENABLE_TESTABILITY=YES`.
2. **Debug and Release modules are incompatible in one derived-data path.**
   Reusing the Debug tests' `-derivedDataPath` produces the same error even
   with testability on. Fix: a dedicated path (e.g. `.xcodebuild-tests-rel`).
3. **Environment variables don't reach the test process** unless prefixed
   `TEST_RUNNER_` (xcodebuild strips the prefix and forwards).

# Examples

The full working invocation:

```bash
TEST_RUNNER_LTX_E2E_LIPDUB=1 \
TEST_RUNNER_LTX_E2E_LIPDUB_LORA=/path/to/ic-lora.safetensors \
xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
  -derivedDataPath .xcodebuild-tests-rel -skipPackagePluginValidation \
  -skipMacroValidation -configuration Release ENABLE_TESTABILITY=YES test \
  -only-testing:LTXVideoTests/LipDubReuseE2ETests
```

`.gitignore` covers `.xcodebuild*/` — `git add -A` once swallowed an entire
derived-data tree into a commit before that pattern was generalized.

# Citations

[1] Discovered take-by-take in the PR #36 E2E runs — see
[the investigation](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md).
[2] [PR #36 validation protocol](/docs/testing/PR36-validation-protocol.md)
