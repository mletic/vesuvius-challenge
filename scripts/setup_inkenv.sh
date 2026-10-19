#!/usr/bin/env bash
# Build the ink-model environment used by this repo, pinned to the versions the results were produced with.
#   - villa @ merge-ink-pipelines 3ea17f5 (koine_machines: ink_9um inference) + scripts/koine_mps.patch
#   - villa @ main 8aa507c (ink-detection/optimized_inference: the 2 µm ResNet-152 model code)
#   - scrollprize/ink_9um (14 checkpoints) and scrollprize/ink_canonical_2um from Hugging Face
# Usage: scripts/setup_inkenv.sh [DIR]   (default ./third_party). Then export INKENV=DIR/inkenv MODELS=DIR/models VILLA=DIR/villa
set -euo pipefail
DIR=$(cd "$(dirname "$0")/.." && pwd)/${1:-third_party}
mkdir -p "$DIR" && cd "$DIR"
if [ ! -d villa-ink ]; then
  git clone -q --filter=blob:limit=2m --sparse https://github.com/ScrollPrize/villa.git villa-ink
  (cd villa-ink && git checkout -q 3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e && git sparse-checkout set ink-detection vesuvius && git apply ../../scripts/koine_mps.patch)
fi
if [ ! -d villa ]; then
  git clone -q --filter=blob:limit=2m --sparse https://github.com/ScrollPrize/villa.git villa
  (cd villa && git checkout -q 8aa507c1215f1fa8758f13d0538f7af9710e02d7 && git sparse-checkout set ink-detection)
fi
[ -d inkenv ] || uv venv -q --python 3.12 inkenv
VIRTUAL_ENV=$DIR/inkenv uv pip install -q torch==2.14.1 torchvision==0.29.1 timm==1.0.30 numpy==2.2.0 scipy==1.18.1 \
  zarr==2.18.7 numcodecs==0.15.1 tifffile==2026.10.15 imagecodecs==2026.10.10 einops==0.8.2 tqdm==4.70.1 \
  fsspec==2026.9.0 aiohttp==3.14.5 requests==2.34.2 pyyaml==6.0.3 huggingface-hub==2.2.0 \
  opencv-python-headless==5.0.0.93 pillow==13.0.0 pynrrd==1.1.3 albumentations
VIRTUAL_ENV=$DIR/inkenv uv pip install -q --no-deps -e villa-ink/vesuvius -e villa-ink/ink-detection
"$DIR/inkenv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
for seed in (42, 43):
    for st in ("010000", "020000", "030000", "040000", "050000", "060000", "075000"):
        hf_hub_download("scrollprize/ink_9um", f"hybrid_3d2d-seed{seed}/step-{st}.pth", local_dir="models/ink_9um")
hf_hub_download("scrollprize/ink_canonical_2um", "r152_3ddec_v2_l5_epoch13.ckpt", local_dir="models/ink_2um")
print("models ok")
PY
echo "export INKENV=$DIR/inkenv MODELS=$DIR/models VILLA=$DIR/villa"
