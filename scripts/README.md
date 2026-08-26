# Scripts

## benchmark.sh — Performance Benchmark Suite

Runs all inference pipelines (T2V, I2V, retake, audio) with fixed seeds and generates a shareable `RESULTS.md`.

### Quick start

```bash
# 1. Build in Release mode
xcodebuild -scheme ltx-video -configuration Release \
    -derivedDataPath .xcodebuild -destination 'platform=macOS' build

# 2. Download models (first time only, ~30GB)
.xcodebuild/Build/Products/Release/ltx-video download

# 3. Run benchmarks
./scripts/benchmark.sh --quick              # ~10 min (9-frame tests only)
./scripts/benchmark.sh                      # ~1 hour (includes 10-second videos)
./scripts/benchmark.sh --quick --skip-audio # ~6 min (no audio model loading)
```

### Output

Results are saved to `benchmarks/<timestamp>/`:

```
benchmarks/20260317_143022/
  RESULTS.md              <- Share this! (paste into GitHub issue)
  benchmark.log           <- Full console log
  t2v-768x512-9f.txt      <- Profiling details per test
  t2v-768x512-9f.mp4      <- Generated video
  i2v-768x512-9f.txt
  i2v-768x512-9f.mp4
  ...
```

### Share your results

Copy `RESULTS.md` to clipboard and create a benchmark issue:

```bash
cat benchmarks/*/RESULTS.md | pbcopy
# Then: https://github.com/VincentGourbin/ltx-video-swift-mlx/issues/new?template=benchmark.yml
```

This helps us track performance across different Apple Silicon chips and memory configurations.

### Options

| Flag | Description |
|------|-------------|
| `--quick` | Only run 9-frame tests (skip 241-frame / 10-second videos) |
| `--skip-audio` | Skip audio pipeline test (requires separate audio model download) |
| `--help` | Show usage |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINARY` | `.xcodebuild/Build/Products/Release/ltx-video` | Path to the CLI binary |
| `SEED` | `42` | Random seed for reproducibility |
| `COOLDOWN` | `120` | Seconds between tests (thermal throttling mitigation) |

### Reference times (M3 Max 96GB)

| Test | Time |
|------|------|
| T2V 768x512 9f | ~33s |
| T2V 1024x576 241f | ~895s (~15 min) |
| I2V 768x512 9f | ~39s |
| I2V 1024x576 241f | ~755s (~12.5 min) |
| Retake 768x512 9f | ~22s |
| Audio I2V 768x512 9f | ~58s |

## package-release.sh — Distributable CLI archive

Builds Release and packages `dist/ltx-video-macos-arm64.zip` for a GitHub release.

```bash
./scripts/package-release.sh v0.3.0
```

The executable **cannot run on its own**: MLX loads its Metal shaders from
`mlx-swift_Cmlx.bundle` at runtime, so the archive ships the binary together with
every resource bundle the build produces. The v0.1.0 archive contained only the
binary and crashed with `Failed to load the default metallib` for everyone who
downloaded it — hence the guards here.

The script aborts rather than produce a broken archive if:

- no resource bundle was built, or `default.metallib` is missing
- the binary is coverage-instrumented (it would drop a `default.profraw` in the
  user's working directory on every run)
- `ltx-video --version` disagrees with the version being packaged — bump
  `LTXVideo.version` and the CLI's `CommandConfiguration.version` first

Publish with:

```bash
gh release create v0.3.0 dist/ltx-video-macos-arm64.zip --title '...' --notes-file NOTES.md
```

## Reference parity scripts

`*_reference.py` run Lightricks' own modules and dump the numbers the Swift port
must reproduce. They exist because these components are numerical: a mis-ported
attention pooler or an off-by-one grid still returns a plausible-looking value,
so eyeballing proves nothing.

| Script | Pins |
| --- | --- |
| `transformer_reference.py` | DiT block forward |
| `diffvae_reference.py` | diffusion decoder |
| `temporal_upscaler_reference.py` | latent temporal upsampler |
| `keyframe_slot_reference.py` | generated keyframe RoPE positions |
| `duration_head_reference.py` | duration head forward + seconds→frames |

All need upstream checked out:

```bash
git clone https://github.com/Lightricks/LTX-2
```

`duration_head_reference.py` needs only `torch` and `safetensors` — it reads
upstream's two grid functions out of `helpers.py` with `ast` rather than
importing them, because `ltx_pipelines.utils` drags in `av` and `OpenImageIO`
for what is integer arithmetic:

```bash
LTX2_ROOT=$PWD/LTX-2 PYTHONPATH=$LTX2_ROOT/packages/ltx-core/src \
  python3 scripts/duration_head_reference.py \
    ~/models/ltx-2.5-duration-head/ltx-2.5-duration-head-bf16.safetensors
```

Its output is pinned by `DurationHeadE2ETests` (forward) and
`DurationGridSnapTests` (seconds→frames, no checkpoint needed).
