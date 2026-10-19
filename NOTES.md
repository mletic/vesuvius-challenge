# Working notes

The record of how the work went, in order. Later sections correct earlier ones; superseded claims are marked
**[superseded]** with a pointer to what replaced them. Current results are in the [README](README.md), the
[write-up](https://mletic.github.io/vesuvius-challenge/) and [PREREG.md](PREREG.md).

Coordinates like `z1892/y3480/x3200` are voxels on level 2 of a scroll's 2.4 µm scan (9.6 µm), the grid the
surface predictions use. Large intermediate files (volumes, predictions, logs) were kept locally and are not in the
repository; the scripts regenerate them.

## 1. Starting point (2026-10-17)

- The open data needs no sign-up: public S3 bucket `vesuvius-challenge-open-data`, CC BY-NC 4.0. Volumes are
  chunked OME-Zarr, so a region can be streamed without downloading a scan. Segments hold a mesh
  (`mesh/*.tifxyz`, plus OBJs) and a surface volume (`surface-volumes/*.zarr`, e.g. 31 layers at 8.64 µm).
- Of the 21 scrolls on the challenge's unread list, only **PHerc0800** had published segments (6). PHerc1203 had
  raw auto-grown meshes only; the rest had volumes and model predictions (`representations/predictions/`). Reading
  them means segmenting first, by hand in VC3D or from the published surface predictions.
- `vesuvius-sweep` renders published segments for visual inspection (mean of the middle layers, CLAHE, 1 cm bar).
  PHerc0800's six segments showed no visible ink.

## 2. Micro-segmentation and its validation (2026-10-17/18)

- `vesuvius_sweep.microseg` and `vesuvius_sweep.grow` build surface volumes from the `m7` surface prediction
  (available for 11 of the 21 unread scrolls): structure-tensor normal at a seed, one sheet tracked as a height field
  over prediction peaks (tile by tile, depth window re-centred, repeated passes), CT sampled along the local normal
  into a 31-layer (9.6 µm) or 65-layer (`--hires`, 2.4 µm) volume in the published layout.
- **Validated at 9 µm.** PHerc0139 w043 (known text): ink from the mesh-free surface correlates r = 0.55 with ink
  from the published segment (0.59 sampling along the published mesh). The tracked sheet sits within ~0.4 voxel
  (median) of the published mesh.
- **Validated at 2.4 µm.** PHerc0841 w00: the 2 µm model on the mesh-free surface correlates r = 0.72 with the same
  model on the published segment, after the −16° rotation between frames (wrong rotations 0.08–0.23). The normal's
  sign must put the papyrus body on +n (`orient_body_positive`), otherwise the output is mirrored.
- **[superseded]** "In-plane orientation matters a lot (r 0.07–0.13 in the wrong orientation)." Those runs also had
  the depth order reversed, which was the real cause (villa #1648). See section 5.
- **Scale.** Letters are ~1.5–3 mm: ~150–300 px at 9 µm, ~600–900 px at 2.4 µm. Small patches of real ink look like
  blobs; text has to be judged over ≥ 1.5 cm.
- **9 µm is not legible.** On known text ink_9um gives ~1 mm blobs in rows; published 2.4 µm predictions of the same
  areas are crisp. Reading letters needs 2.4 µm data: of the unread scrolls, PHerc1203 and PHerc0846A have 2.4 µm
  scans with surface predictions.
- **Paris 4 is a poor control** for this method: tightly packed sheets, the tracker hops windings, and its published
  meshes sit between surface-prediction peaks.
- **PHerc0800:** no letter-like ink at 9 µm in either orientation; one persistent bright band follows a CT crease.

## 3. First searches at 2.4 µm (2026-10-18)

- **PHerc1203**, 15 hand-picked seeds (3 heights × 5 clean-sheet regions), 5 mm patches, 2 µm model. Ink fraction
  0.004–0.14 against 0.31 on known text (PHerc0841). The two highest sit on cracks and air gaps. The cleanest
  candidate (z1892/y3480/x3200), grown to 1 cm: ink fraction 0.16, scattered blocky blobs, no lines or letters, not
  explained by CT dark spots (correlation −0.17; control −0.12).
- **PHerc0846A**, 15 seeds. Four patches at or above the known-text fraction (0.55, 0.51, 0.40, 0.36), not tracking
  CT gaps. The top two are the same place (y≈2800–3120, x≈3600–4000) at two heights. Letter-like "ΙΠ" marks in the CT
  are ~0.35 mm fibre strips bridging an air gap (too small; they change with depth). Grown to 1 cm: 46% of the area
  saturated in merged clouds with tile-shaped edges, no discrete strokes: model over-response, not writing.

## 4. Text-structure score and wider triage (2026-10-19)

- **[superseded]** v1 "peakness" score (`textscore.line_score`): line-band power against its flanks, vertical against
  horizontal. Early calibration looked convincing (known text +1.6..+1.8, presumed blank −0.9..+0.35); a plain
  anisotropy score had failed because fibres and tears are horizontal too. The pre-registered benchmark (section 6)
  shows text scores do not separate text from no text on 9 µm outputs.
- `scripts/triage.py`: automatic seeds on surface-prediction sheets, 1.5 cm fields grown at 9.6 µm, ink_9um, score.
  33 fields scored; four reached the "known-text" range and all four failed inspection (torn surfaces near the ends
  of the scanned band; a field whose own outline produced the periodicity). Added guards: the score of the field's
  hole/outline map (`mask_peak`), coverage and hole fraction.

## 5. Orientation, test-time augmentation, audit (2026-10-19)

- `scripts/orientation_sensitivity.py`, against the organisers' labels on PHerc0841 w00: ink_9um gives AUC 0.79–0.85
  in all 8 in-plane rotations/flips with the published depth order, 0.49–0.65 with depth reversed. The 2 µm model
  gives 0.988–0.990 in the four orientations run.
- Averaging all 8 orientations (0.849) adds nothing over the existing `--tta-mirror` (0.849), which is +0.03 over a
  single pass (0.816).
- `scripts/audit_orientation.py`: of 908 published meshes (308 segments), 500 run the scroll axis down the rows and
  408 up (all of Paris 4 and PHerc0172, half of PHerc1667). Harmless for ink_9um given the result above.

## 6. Pre-registered benchmark and checkpoint pilot (2026-10-19)

Plan in [PREREG.md](PREREG.md); raw scores in [results/](results/).

- Line spacing from labels: median 6.55 mm (P10–P90 5.65–7.27 mm, 73 measurements on 13 segments).
- v2 (red-noise) and v3 (row-profile) scores were designed and fixed from unit tests on synthetic data before any
  benchmark scoring (PREREG deviations 2–4). v3 is primary.
- P3 passed: published 2.4 µm predictions vs their 90°-rotated copies, AUC 0.916 (0.973 label-confirmed).
- P1/P2 failed: on ink_9um outputs of 14 labelled segments, v3 picks the text direction 6/14, tile AUC 0.502.
- The separation rule from villa #1867 picks the depth direction 14/14, and 14/14 checkpoints in the pilot.
- No label-free score ranks the 14 ink_9um checkpoints on PHerc0841 w00 (Spearman +0.002, +0.011, −0.06).
- Consequence: the triage scores of section 4 could not have flagged text; the PHerc1203 and PHerc0846A negatives
  rest on the 2.4 µm reads of section 3.
