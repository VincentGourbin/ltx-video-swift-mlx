#!/bin/bash
# package-release.sh — Build and package a distributable ltx-video CLI archive.
#
# The executable cannot run on its own: MLX loads its Metal shaders from
# mlx-swift_Cmlx.bundle at runtime. Shipping the binary alone produces a
# "Failed to load the default metallib" crash on the user's machine — which is
# exactly what happened to the v0.1.0 archive. This script always packages the
# resource bundles alongside the binary and refuses to produce an archive that
# would be broken on arrival.
#
# Usage:
#   ./scripts/package-release.sh v0.3.0
#
# Output:
#   dist/ltx-video-macos-arm64.zip

set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "usage: $0 <version>   (e.g. $0 v0.3.0)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR=".xcodebuild"
PRODUCTS="$BUILD_DIR/Build/Products/Release"
STAGE="dist/ltx-video-macos-arm64"
ARCHIVE="dist/ltx-video-macos-arm64.zip"

echo "==> Building Release ($VERSION)"
# -skipPackagePluginValidation / -skipMacroValidation: mlx-swift ships a build
#   plugin and mlx-swift-lm ships macros; both need interactive approval
#   otherwise, which is unavailable in a non-interactive build.
# *_CODE_COVERAGE=NO: without these the linked binary is instrumented and drops
#   a default.profraw in the user's working directory on every run.
xcodebuild \
    -scheme ltx-video \
    -configuration Release \
    -derivedDataPath "$BUILD_DIR" \
    -destination 'platform=macOS,arch=arm64' \
    -skipPackagePluginValidation \
    -skipMacroValidation \
    ENABLE_CODE_COVERAGE=NO \
    CLANG_ENABLE_CODE_COVERAGE=NO \
    SWIFT_ENABLE_CODE_COVERAGE=NO \
    build

echo "==> Staging"
rm -rf "$STAGE" "$ARCHIVE"
mkdir -p "$STAGE"
cp "$PRODUCTS/ltx-video" "$STAGE/"

# Every resource bundle the build produces, not just the Metal one — a missing
# bundle only ever surfaces at runtime, so copy them all rather than guess.
shopt -s nullglob
bundles=("$PRODUCTS"/*.bundle)
shopt -u nullglob
if [ ${#bundles[@]} -eq 0 ]; then
    echo "FAIL: no resource bundles in $PRODUCTS" >&2
    exit 1
fi
cp -R "${bundles[@]}" "$STAGE/"

METALLIB="$STAGE/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [ ! -f "$METALLIB" ]; then
    echo "FAIL: $METALLIB missing — the archive would crash on first MLX call." >&2
    exit 1
fi
echo "    default.metallib: $(du -h "$METALLIB" | cut -f1)"

echo "==> Smoke test (from the staged directory, as a user would run it)"
smoke_dir="$(cd "$STAGE" && pwd)"
(
    cd "$smoke_dir"
    ./ltx-video --version
    ./ltx-video models >/dev/null
    if [ -e default.profraw ]; then
        echo "FAIL: binary is coverage-instrumented (wrote default.profraw)." >&2
        exit 1
    fi
)

reported="$(cd "$smoke_dir" && ./ltx-video --version)"
if [ "$reported" != "${VERSION#v}" ]; then
    echo "FAIL: binary reports $reported but packaging $VERSION." >&2
    echo "      Bump LTXVideo.version and CommandConfiguration.version first." >&2
    exit 1
fi

echo "==> Archiving"
(cd dist && zip -qry "$(basename "$ARCHIVE")" "$(basename "$STAGE")")
echo "    $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"

echo
echo "Done. Publish with:"
echo "  gh release create $VERSION $ARCHIVE --title '...' --notes-file ..."
