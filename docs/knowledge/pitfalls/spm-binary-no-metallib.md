---
type: Pitfall
title: swift build binaries crash at MLX runtime (metallib not found)
description: The SPM-built CLI compiles fine but aborts on first GPU op; only xcodebuild products carry the MLX metallib bundle correctly.
tags: [mlx-swift, build, metal, cli]
timestamp: 2026-07-16T00:00:00Z
---

A `swift build` binary of the CLI compiles and links, then aborts on the first
MLX GPU operation with:

```
MLX error: Failed to load the default metallib. library not found ...
  at .../mlx-c/mlx/c/stream.cpp:115
```

`swift test` hits the same wall for any MLX-touching test. Running the binary
from its own build directory does NOT fix it (verified 2026-07-16, twice).

# The defense

- **Syntax/type checking**: `swift build` is fine (and fast) — use it for that.
- **Running the CLI**: build via xcodebuild and run from the products dir
  (the metallib bundle must sit alongside the binary):

```bash
xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild \
  -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation build
cd .xcodebuild/Build/Products/Release && ./ltx-video ...
```

- **Running tests**: `xcodebuild ... test` (the xctest bundle resolves the
  metallib; the full suite including MLX tests passes there). For Release-mode
  tests see [the testability pitfall](/docs/knowledge/pitfalls/release-tests-need-testability.md).

# Citations

[1] Validated live in the PR #36 campaign — see
[the investigation](/docs/knowledge/investigations/lipdub-segmentation-asks-2026-07.md).
