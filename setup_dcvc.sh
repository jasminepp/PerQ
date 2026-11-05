#!/bin/bash
# Fetch DCVC-RT (Jia et al., "Towards Practical Real-Time Neural Video Compression", CVPR 2025) at the pinned commit,
# apply our patch and build its extensions: the rANS entropy coder (required) and, with --cuda, the CUDA inference
# kernels (optional; faster decoding).
# The patch puts the rANS coder in a dcvc namespace and splits a templated comparison operator into its float4 and
# Half4 overloads.
set -e
cd "$(dirname "$0")"

DCVC_COMMIT=447d761c9ec9affd93edb49c0c29380a9b80081b
DCVC_DIR=third_party/DCVC
PATCH=$PWD/third_party/dcvc-rt_447d761.patch

if [ ! -d "$DCVC_DIR" ]; then
    git clone https://github.com/microsoft/DCVC.git "$DCVC_DIR"
fi
git -C "$DCVC_DIR" checkout -q "$DCVC_COMMIT"
if git -C "$DCVC_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "== patch already applied"
else
    git -C "$DCVC_DIR" apply "$PATCH"
fi

echo "== entropy coder (MLCodec_extensions_cpp)"
pip install "./$DCVC_DIR/src/cpp" --no-build-isolation

if [ "$1" = "--cuda" ]; then
    echo "== CUDA inference kernels (inference_extensions_cuda)"
    pip install "./$DCVC_DIR/src/layers/extensions/inference" --no-build-isolation
fi

python -c "import MLCodec_extensions_cpp; print('entropy coder ok')"