# Vesuvius Challenge

Working repo for the [Vesuvius Challenge](https://scrollprize.org): reading carbonised Herculaneum scrolls from X-ray CT.

## Data

- No sign-up needed. The data sits in the public S3 bucket `vesuvius-challenge-open-data` (browse it [here](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html)) under CC-BY-NC 4.0.
- Volumes are OME-Zarr, chunked and multi-resolution, so you can stream just the region you need.
- Segments contain:
  - a mesh: `mesh/*.tifxyz` (x/y/z TIFFs) plus OBJs
  - a surface volume: `surface-volumes/*.zarr`, 31 layers × H × W, uint8, 8.64 µm/px

**Finding:** as of today, only **PHerc0800** has published segments (6) among the 21 scrolls on the challenge's unread list. PHerc1203 has `segments/raw/` only. The rest have just volumes plus model predictions (`representations/predictions/{surfaces,lasagna}`). So reading these scrolls means segmenting it yourself in [VC3D](https://scrollprize.org/tutorial_VC3D), or building surfaces from the published surface predictions.

## Tool: `vesuvius-sweep`

The tool streams segment surface volumes for unread scrolls and saves flattened renders: the mean of the middle layers, with CLAHE contrast and a 1 cm scale bar. You then look through them by eye for visible ink.

```bash
uv run vesuvius-sweep --list              # segment inventory for all unread scrolls
uv run vesuvius-sweep PHerc0800 --level 0 # full-res renders -> renders/PHerc0800/*.png
```

Options: `--band N` sets how many middle layers are averaged, and `--level` picks the pyramid level (0 = full res). If a fetch fails with "Server disconnected", re-run; finished renders are skipped.

## Micro-segmentation: no hand-traced mesh needed

`vesuvius_sweep.microseg` / `vesuvius_sweep.grow` build surface volumes for any scroll that has a published
`m7` surface prediction (11 of the 21 scrolls on the challenge's unread list, plus controls):

1. take the sheet normal from the structure tensor of the surface prediction around a seed;
2. track one sheet as a height field over surface-prediction peaks (best-first growth; `grow` does it tile by tile,
   re-centring each tile's depth window so it follows a curving sheet, with repeated passes);
3. sample the CT along the local normal → a 31-layer (9 µm) or 65-layer (`--hires`, 2.4 µm) surface volume in the
   **published layout** (scroll axis down the rows, layer index running from the papyrus body outward).

```bash
uv run python -m vesuvius_sweep.microseg PHerc1203 1892 3480 3200 --size 512        # one 512² patch (level-2 coords)
uv run python -m vesuvius_sweep.grow PHerc0139-9um 6559 4338 4132 --size 1536        # tiled, larger
uv run python -m vesuvius_sweep.grow PHerc0841 1984 3997 3553 --size 512 --hires --layers 65   # 2.4 µm output
INKENV=... MODELS=... scripts/ink_patch.sh data/microseg/<name>.zarr                 # ink_9um, 2 ckpts × 3 windows
python scripts/infer_r152.py <zarr> <ink_canonical_2um ckpt> out.tif --villa <villa>  # 2 µm model on MPS
```

Ink models run locally on Apple MPS. I patched the device choice in the `merge-ink-pipelines` branch's
`koine_machines` inference; upstream `main` already runs `vesuvius.ink_detection` on MPS (villa #1865), so that
patch is not needed there.

## Findings so far (2026-10-18)

- **Validated at 9 µm.** At PHerc0139 segment w043 (known text), ink predicted from *my* micro-segment correlates
  r = 0.55 with ink predicted from the published hand-made segment (r = 0.59 when I sample along the published mesh
  instead of my tracked surface). The tracked sheet sits within ~0.4 voxel (median) of the published mesh.
- **Depth order matters; in-plane orientation barely does.** *(Corrected 2026-10-19.)* An earlier version of this
  README said in-plane orientation "matters a lot" (r 0.07–0.13 in the wrong orientation). Those runs also had the
  depth order reversed, which is the real cause (already documented upstream in villa #1648). Measured cleanly
  against the organisers' labels on PHerc0841 w00 (`scripts/orientation_sensitivity.py`): ink_9um gives AUC
  0.79–0.85 in all 8 in-plane rotations/flips with the published depth order, and 0.49–0.65 with depth reversed;
  the 2 µm model gives 0.988–0.990 in the four orientations run. Averaging the 8 orientations (0.849) adds nothing
  over the existing `--tta-mirror` (0.849), which is +0.03 over a single pass (0.816).
- **Published segments run the scroll axis both ways** (`scripts/audit_orientation.py`, 908 meshes on 308
  segments): 500 rows-down, 408 rows-up (all of Paris 4 and PHerc0172, half of PHerc1667). Given the result above
  this is harmless for ink_9um; it matters only for anyone comparing renders pixel by pixel.
- **Scale calibration.** Letters are ~1.5–3 mm: ~150–300 px at 9 µm, ~600–900 px at 2.4 µm. A 512 px patch at
  9 µm holds only a few letters, and real ink there *looks* like blobs; judge text over ≥1.5 cm fields.
- **9 µm is not legible.** On known text the 9 µm model gives ~1 mm blobs in rows; the published 2.4 µm predictions
  of the same areas are crisp. Reading letters credibly needs 2.4 µm data → PHerc1203 and PHerc0846A
  (both unread, both have 2.4 µm scans + m7-L2 surface predictions).
- **Validated at 2.4 µm.** At PHerc0841 segment w00 (known text, mesh and surface prediction agree), the 2 µm
  ResNet-152 model on *my* `grow --hires` micro-segment correlates r = 0.72 with the same model on the published
  segment, after the −16° in-plane rotation between the two frames (wrong rotations: 0.08–0.23). The normal's sign
  is fixed so the papyrus body lies on +n (`orient_body_positive`); without that the output came out mirrored.
- **Paris 4 is a bad control** for this method: sheets are tightly packed, the tracker hops windings, and the
  published meshes sit between surface-prediction peaks (cf. villa issues on transforms not fitting landmarks).
- **PHerc0800 published segments** (the only published segments on an unread scroll): no letter-like ink at
  9 µm in either orientation; a persistent bright band coincides with a CT crease (artifact).

## Widened hunt with a text-structure score (2026-10-19)

- `textscore.line_score`: in published layout text lines are horizontal and repeat every ~4–7 mm. A pure
  anisotropy score failed (fibres and tears are horizontal too: presumed-blank PHerc0800 scored like text). The
  **peakness** score (power in the line-period band vs its flanks, vertical vs horizontal) separates: known text
  +1.6..+1.8 (my w043 micro-seg +1.79, published +1.60, published 2.4 µm maps in 2 cm tiles median +1.5..+1.9,
  margins low), presumed blank −0.9..+0.35.
- Two-stage hunt: `scripts/triage.py` auto-seeds sheets (5 heights, ≥800 L2 vox apart), grows 1.5 cm fields at
  9.6 µm, runs ink_9um, logs scores (`data/triage_<scroll>.csv`); then 2.4 µm only for the best.
- 35 seeds → 33 scored. Four reached ≥ +1.5, none survived inspection: both PHerc1203 hits are torn, frayed
  surfaces near the ends of the scanned band (ink fires on fragment edges); the top PHerc0846A field's *mask*
  alone scores +2.34 (outline geometry drives the periodicity), and its 2.4 µm 1 cm read has 6% sparse blobs,
  several beside holes, no strokes.
- Lesson: score the mask too (`mask-peak`) and require clean coverage before trusting a high ink score.

## Pre-registered benchmark of the text score (2026-10-19)

Plan in `PREREG.md` (committed before the runs; five dated deviations, all made before benchmark scoring except a
disclosed exploratory look). Results in `results/`; reproduce with `make env && make benchmark`.

- **Line spacing from labels:** median 6.55 mm (P10–P90 5.65–7.27 mm, 73 measurements on 13 segments).
- **P3 passed:** on the 28 labelled segments' published 2.4 µm predictions, the primary score (v3, row-profile
  periodicity) tells text tiles from their 90°-rotated copies at AUC 0.916 (1,701 tiles; 0.973 on 535 label-confirmed).
- **P1/P2 failed:** on ink_9um outputs of 14 labelled segments (5 scrolls), with the same map in the wrong depth
  direction as the negative, v3 picks the text direction 6/14 and reaches tile AUC 0.502 (0.568 on label-confirmed
  tiles, exploratory). At 9 µm ink is blobby and a 20 mm tile holds ~3 line periods. Injected synthetic lines are
  detected only when stronger than real ink.
- **The separation rule (villa #1867) picks the depth direction 14/14** here, and 14/14 checkpoints in the pilot.
- **Checkpoint ranking:** no label-free score ranks the 14 ink_9um checkpoints on held-out PHerc0841 w00
  (Spearman: text +0.002, separation +0.011, v3 −0.06).
- **Consequence for the search:** the 9 µm triage scores could not have flagged text; the PHerc1203/PHerc0846A
  negatives rest on the 2.4 µm reads.

## Next steps

1. PHerc1203 hunt done (13/15 seeds produced patches; ink fraction 0.004–0.14 vs 0.31 on known text). Two top
   patches' blobs sit on cracks/air gaps (artifacts); z1892/y3480/x3200 is clean with letter-sized blobs in rough
   rows. Grown to 1 cm (`_S1024_hires`, 79% coverage): ink fraction 0.16, scattered blocky blobs with no line or
   letter structure; not explained by CT dark spots (corr −0.17, control −0.12). Verdict: no legible text there. Original hunt: `scripts/hunt.sh PHerc1203 ...` over 15 seeds (3 heights × 5 clean-sheet regions), log in
   `data/hunt_1203.log`, previews `data/grow/*_view.png`. Reference: on known text (PHerc0841) the 2 µm model gives
   ~31% of pixels > 0.5.
2. PHerc0846A hunt done (14/15 patches; log `data/hunt_0846A.log`). Four patches score at/above the known-text
   control (0.55, 0.51, 0.40, 0.36 vs 0.31); scores do not track CT gaps (corr with darkness ≈ 0, like the
   control). The top two are the same x/y (y≈2800–3120, x≈3600–4000, L2) at z946 and z1892. At 5 mm the maps are
   broad saturated regions, not strokes; letter-like "ΙΠ" marks in the CT are ~0.35 mm fibre strips bridging an air
   gap (too small, change with depth). Grown to 1 cm (89% coverage): 46% of the area saturated in merged clouds with tile-shaped edges, no discrete
   letters or lines → model over-response on this papyrus, not legible text. Then widen sampling; then grow promising patches to ≥ 2 cm × 2 cm (about 4 cm²) and render at full res.
3. Write-up of the method, controls and negative results (MPS support is already upstream).
4. Do the in-browser ink-labelling exercise ([get_started](https://scrollprize.org/get_started)) and join the Discord.
