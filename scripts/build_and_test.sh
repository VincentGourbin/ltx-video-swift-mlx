#!/bin/bash
# Build and test LTX-Video-Swift-MLX
# Usage: ./scripts/build_and_test.sh [component]
# Components: transformer, vae, connector, all, validate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
XCODE_BUILD_DIR="$PROJECT_DIR/.xcodebuild"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LTX-Video-Swift-MLX Build & Test ===${NC}"
echo "Project: $PROJECT_DIR"
echo ""

# Build with Xcode (required for Metal support)
echo -e "${YELLOW}Building with xcodebuild...${NC}"
cd "$PROJECT_DIR"

# Build using xcodebuild (required for Metal/MLX to work properly)
xcodebuild -scheme ltx-video \
    -configuration Release \
    -destination "platform=macOS,arch=arm64" \
    -derivedDataPath "$XCODE_BUILD_DIR" \
    build 2>&1 | grep -E "(BUILD|Compiling|Linking|error:|warning:)" | tail -30

# Find the built executable
EXECUTABLE=$(find "$XCODE_BUILD_DIR/Build/Products/Release" -name "ltx-video" -type f -perm +111 2>/dev/null | head -1)

if [ -z "$EXECUTABLE" ]; then
    echo -e "${YELLOW}Looking for executable in alternate locations...${NC}"
    EXECUTABLE=$(find "$XCODE_BUILD_DIR" -name "ltx-video" -type f -perm +111 2>/dev/null | head -1)
fi

if [ -z "$EXECUTABLE" ]; then
    echo -e "${RED}Build failed: executable not found${NC}"
    echo "Try building manually in Xcode first"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"
echo "Executable: $EXECUTABLE"
echo ""

# Run tests based on component
COMPONENT="${1:-all}"

case "$COMPONENT" in
    transformer)
        echo -e "${YELLOW}Testing Transformer...${NC}"
        "$EXECUTABLE" test --component transformer --debug
        ;;
    vae)
        echo -e "${YELLOW}Testing VAE...${NC}"
        "$EXECUTABLE" test --component vae --debug
        ;;
    connector|text-encoder)
        echo -e "${YELLOW}Testing Text Encoder Connector...${NC}"
        "$EXECUTABLE" test --component connector --debug
        ;;
    validate)
        echo -e "${YELLOW}Running Validation Tests...${NC}"
        "$EXECUTABLE" validate --debug
        ;;
    all)
        echo -e "${YELLOW}Testing All Components...${NC}"
        "$EXECUTABLE" test --component all --debug
        ;;
    *)
        echo "Usage: $0 [transformer|vae|connector|validate|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
