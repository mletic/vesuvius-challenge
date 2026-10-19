# Mesh-free ink search for the Herculaneum scrolls

Tools and a pre-registered benchmark for finding ink in CT scans of the
[Vesuvius Challenge](https://scrollprize.org) scrolls without hand-traced meshes. Surface volumes are grown directly
from the organisers' published surface predictions, the published ink models run on them locally, and label-free
diagnostics of the resulting ink maps are tested against the organisers' ink labels.

**Write-up with figures:** [mletic.github.io/vesuvius-challenge](https://mletic.github.io/vesuvius-challenge/) ·
**Plan and deviations:** [PREREG.md](PREREG.md) · **Raw scores:** [results/](results/) ·
**Working notes:** [NOTES.md](NOTES.md)

## Key results

- **Surfaces without a mesh match hand-traced ones.** On known text, ink predicted on the mesh-free surfaces
  correlates with ink predicted on the published segments at r = 0.55 (9 µm, PHerc0139 w043) and r = 0.72
  (2.4 µm, PHerc0841 w00).
- **A line-periodicity text score works on crisp maps only.** Lines of writing are 6.55 mm apart (median, measured
  from the labels). The score separates published 2.4 µm predictions from their rotated copies (AUC 0.916; 0.973
  where labels confirm ink) but is at chance on 9 µm `ink_9um` outputs (tile AUC 0.502), even where the model reads
  the text almost perfectly.
- **The separation rule from villa #1867 picks the depth direction reliably:** 14 of 14 labelled segments across
  five scrolls, and 14 of 14 checkpoints on held-out PHerc0841.
- **No label-free score ranks checkpoints** on held-out PHerc0841 (Spearman +0.01 for separation, +0.00 and −0.06
  for the text scores).
- **In-plane orientation barely matters to `ink_9um`** (AUC 0.79–0.85 over all 8 rotations and flips); reversing the
  depth order does (0.49–0.65). Published meshes run the scroll axis both ways (500 rows-down, 408 rows-up).
- **No text found** in automatically grown surfaces on two unread scrolls, PHerc1203 and PHerc0846A, at 2.4 µm.

## Quick start

Needs [uv](https://docs.astral.sh/uv/), Python 3.12, git, and an Apple Silicon (MPS) or CUDA GPU for inference.
The benchmark downloads about 15 GB and takes about 6 hours on an M1 Pro.

```bash
make env         # pinned ink-model environment and weights (once); then export the variables it prints
make test        # unit tests of the text scores
make benchmark   # label inventory, two-direction inference on 14 segments, all pre-registered outcomes
make report      # write-up page: report/index.html and docs/index.html
```

Grow a surface volume and score an ink map directly:

```bash
uv run python -m vesuvius_sweep.grow PHerc0841 1984 3997 3553 --size 512 --hires --layers 65   # 2.4 µm, no mesh
uv run python scripts/text_line_score.py INK.tif --um 2.4 --out heat.png                    # text score of any ink map
```

## What is in the repository

| Path | What it does |
|---|---|
| `src/vesuvius_sweep/` | The library: mesh-free surface volumes (`microseg.py`, `grow.py`), text scores (`textscore.py`), bucket access and render helpers |
| `scripts/setup_inkenv.sh`, `koine_mps.patch` | Pinned environment for the published ink models, with Apple Silicon support |
| `scripts/infer_r152.py` | The 2 µm ResNet-152 ink model on MPS |
| `scripts/text_line_score.py` | Text score of any ink map, with heat overlay and geometry guard |
| `scripts/benchmark/` | The pre-registered benchmark: label inventory, line spacing, data, inference, evaluation, checkpoint pilot |
| `scripts/experiments/` | Orientation experiments, the audit of published meshes, positive controls against published segments |
| `scripts/search/` | Search tools: automatic seeding and triage, 2.4 µm reads, segment sweeps |
| `scripts/writeup/`, `docs/` | The write-up page (`docs/` is served by GitHub Pages) |
| `results/` | Raw scores behind every number in the write-up (see `results/README.md`) |
| `tests/` | Unit tests of the scores on synthetic lines and noise |

## How it works

**Surfaces without a mesh.** Inside a few millimetres, papyrus sheets are close to parallel. A seed's normal comes
from the structure tensor of the published `m7` surface prediction; one sheet is followed as a height field over
prediction peaks, tile by tile, re-centring the depth window as the sheet curves. The CT is sampled along the local
normal into a 31-layer (9.6 µm) or 65-layer (2.4 µm) surface volume with the scroll axis down the rows and the
papyrus body on the same side as in published segments, so the published ink models apply unchanged.

**Text score.** Lines of writing span a tile, so each row is averaged across the tile before looking for the line
period (5.65–7.27 mm, from the labels). The fraction of row-profile power in that band is compared with the same for
columns, over small tilts. Guards reject fields whose holes or outline alone produce the periodicity.

**Benchmark.** Running `ink_9um` with the depth order reversed removes the ink signal but keeps every crack, tear and
fibre, so each surface supplies its own negative. The plan, targets and thresholds were committed before the runs;
every later change is a dated deviation in [PREREG.md](PREREG.md).

## Limitations

- The text score does not work on 9 µm model outputs; it is useful only on crisp, high-resolution ink maps.
- The surface tracker follows one sheet as a height field; it fails on tightly packed or strongly curved regions
  (Paris 4) and tears in damaged areas.
- The searches covered a few percent of each scroll's scanned band with models that transfer poorly to new scrolls
  (villa #1867); the negatives say these areas show no text to the current public models, not that there is none.

## Data, licence and credits

Code: MIT ([LICENSE](LICENSE)). Data: the Vesuvius Challenge open data (CC BY-NC 4.0; EduceLab scans under the same
terms). Models: `scrollprize/ink_9um` and `scrollprize/ink_canonical_2um`; inference code from
[ScrollPrize/villa](https://github.com/ScrollPrize/villa). The separation rule is from villa issue #1867
(@AndreasHad04); the depth-order effect was documented in villa issue #1648.
