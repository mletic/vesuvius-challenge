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

Ink models run locally on Apple MPS (a one-line device patch to villa's `koine_machines` inference, which only
knew CUDA/CPU).

## Findings so far (2026-10-18)

- **Validated at 9 µm.** At PHerc0139 segment w043 (known text), ink predicted from *my* micro-segment correlates
  r = 0.55 with ink predicted from the published hand-made segment (r = 0.59 when I sample along the published mesh
  instead of my tracked surface). The tracked sheet sits within ~0.4 voxel (median) of the published mesh.
- **Orientation matters a lot.** In the wrong in-plane / layer orientation the same model output correlates
  0.07–0.13 with the published one. Published segments are not consistent among themselves: PHerc0800 and Paris 4
  segments run the scroll axis rows-*up*, w043 rows-down. Measure it from the tifxyz before trusting a prediction.
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
3. Upstream the MPS fix (and anything else learned) to villa.
4. Do the in-browser ink-labelling exercise ([get_started](https://scrollprize.org/get_started)) and join the Discord.
