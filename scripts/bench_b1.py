"""PREREG B1: download the 14 labelled ~9.4 µm segment volumes and run ink_9um both directions.

Outputs data/bench/b1/<scroll>_<segment>/{vol.zarr,pred_forward.tif,pred_reverse.tif}.
Waits for the checkpoint-ranking pilot to finish before using the GPU.

usage: INKENV=... MODELS=... uv run python scripts/bench_b1.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

INV = json.load(open("data/bench/vol_inventory.json"))
LAB = json.load(open("data/label_inventory.json"))
OUT = Path("data/bench/b1")
CKPT = "hybrid_3d2d-seed42/step-075000.pth"


def segments():
    seg_paths = {o["segment"].split("/")[2][:40]: o["segment"] for o in LAB}
    for scroll, seg40, vols, shape in INV:
        if vols:
            yield scroll, seg_paths[seg40], vols[0], shape


def pilot_running():
    return subprocess.run(["pgrep", "-f", "scripts/ckpt_rank.py"], capture_output=True).returncode == 0


def main():
    inkenv, models = os.environ["INKENV"], os.environ["MODELS"]
    jobs = list(segments())
    for scroll, seg, vol, shape in jobs:  # downloads first (network only)
        d = OUT / f"{scroll}_{seg.split('/')[2]}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "vol.zarr" / "0" / ".zarray").exists():
            r = subprocess.run([sys.executable, "-m", "vesuvius_sweep.crop", f"{seg}surface-volumes/{vol}", str(d / "vol.zarr"),
                                "--rows", "0", str(shape[1]), "--cols", "0", str(shape[2])], capture_output=True, text=True)
            print(f"download {d.name}: {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr[-200:]}", flush=True)
    while pilot_running():
        time.sleep(60)
    for scroll, seg, vol, shape in jobs:
        d = OUT / f"{scroll}_{seg.split('/')[2]}"
        for direction in ("forward", "reverse"):
            out = d / f"pred_{direction}.tif"
            if out.exists() or not (d / "vol.zarr").exists():
                continue
            t = time.time()
            subprocess.run([f"{inkenv}/bin/python", "-m", "koine_machines.inference.infer", str(d / "vol.zarr"),
                            f"{models}/ink_9um/{CKPT}", str(out), "--no-compile", "--batch-size", "16",
                            "--direction", direction], capture_output=True)
            print(f"infer {d.name} {direction}: {'ok' if out.exists() else 'FAILED'} ({(time.time() - t) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
