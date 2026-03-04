#!/bin/bash
# Generate all 8 flying-car I2V + Audio comparison cases
# Input: docs/examples/example_input.png
# All videos: 121 frames (5s), seed 42, enhanced prompt, audio enabled
#
# Estimated times (Apple Silicon M3 Max 96GB):
# Cases 1,3,4,6,8: ~5-10 min each
# Case 2: ~50 min (40 steps + CFG)
# Case 5: ~15 min (40 steps + upscaler)
# Case 7: ~90 min (40 steps at 1024x576)

set -e
cd "$(dirname "$0")/../../.."

BIN=".xcodebuild/Build/Products/Release/ltx-video"
IMG="docs/examples/example_input.png"
OUT="docs/examples/flying-car"
PROMPT="make this car fly"
COMMON="--image $IMG --audio --enhance-prompt --seed 42 --profile -f 121"

echo "=== Case 1: Distilled ==="
$BIN generate $COMMON -m distilled -w 768 -h 512 \
    -o $OUT/distilled.mp4 "$PROMPT"

echo ""
echo "=== Case 2: Dev ==="
$BIN generate $COMMON -m dev -w 768 -h 512 \
    -o $OUT/dev.mp4 "$PROMPT"

echo ""
echo "=== Case 3: Dev + LoRA ==="
$BIN generate $COMMON --distilled-lora -w 768 -h 512 \
    -o $OUT/dev-lora.mp4 "$PROMPT"

echo ""
echo "=== Case 4: Distilled + Upscaler ==="
$BIN generate $COMMON -m distilled --two-stage -w 768 -h 512 \
    -o $OUT/distilled-upscaler.mp4 "$PROMPT"

echo ""
echo "=== Case 5: Dev + Upscaler (not recommended) ==="
$BIN generate $COMMON -m dev --two-stage --steps 40 --guidance 4.0 -w 768 -h 512 \
    -o $OUT/dev-upscaler.mp4 "$PROMPT"

echo ""
echo "=== Case 6: Dev + LoRA + Upscaler ==="
$BIN generate $COMMON --distilled-lora --two-stage -w 768 -h 512 \
    -o $OUT/dev-lora-upscaler.mp4 "$PROMPT"

echo ""
echo "=== Case 7: Dev 1024x576 ==="
$BIN generate $COMMON -m dev -w 1024 -h 576 \
    -o $OUT/dev-1024x576.mp4 "$PROMPT"

echo ""
echo "=== Case 8: Distilled qint8 ==="
$BIN generate $COMMON -m distilled --transformer-quant qint8 -w 768 -h 512 \
    -o $OUT/distilled-qint8.mp4 "$PROMPT"

echo ""
echo "=== Extracting frame 12 from all videos ==="
mkdir -p $OUT/frames
for f in distilled dev dev-lora distilled-upscaler dev-upscaler dev-lora-upscaler dev-1024x576 distilled-qint8; do
    if [ -f "$OUT/$f.mp4" ]; then
        ffmpeg -y -i "$OUT/$f.mp4" -vf "select=eq(n\,12)" -vframes 1 "$OUT/frames/${f}_frame_12.png" 2>/dev/null
        echo "  Extracted frame 12: $f"
    fi
done

echo ""
echo "=== All cases complete ==="
