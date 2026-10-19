# Pre-registration: label-free text diagnostic

Written 2026-10-19, before any of the runs below. Committed to git so the order is checkable.
Changes after this commit go in the "Deviations" section at the end, with the reason.

## Question

Can a label-free score computed on an ink-probability map tell whether the map shows lines of writing,
well enough to (a) choose the correct depth direction, (b) flag text-bearing areas, and (c) say how much
ink signal a null result would have missed?

## Data

**B1 — natural negatives (primary).** The 14 labelled segments that have a native ~9.4 µm surface volume
(`data/bench/vol_inventory.json`): PHerc0139 ×8 (w030, w035, w039, w040, w041, w043, w044, w045),
PHerc0841 ×3 (w00 and the two auto_grown), PHerc0814 46527, PHerc0009B auto_grown, PHerc0500P2 --1.
For each: `scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000`, default 17-layer centred window,
`--direction forward` and `--direction reverse`, no TTA.

- Labels: organisers' `inklabels` / `supervision` (2.4 µm grid, level 2), box-resampled to the segment's
  9.4 µm canvas; one integer shift in ±16 px chosen per segment by maximising the mean AUC of the two
  directions (applied identically to both, so it cannot favour either).
- **Text direction** (ground truth) = the direction with the higher label AUC. A segment is *ambiguous* and
  excluded from B1 if the two AUCs differ by < 0.05 or both are < 0.60.
- Text-direction maps are positives, wrong-direction maps negatives. Same surface, same cracks and tears,
  same model; only the presence of an ink signal differs.

**B2 — published 2.4 µm predictions.** The 28 labelled segments' published `ink-detection` canonical
predictions, block-averaged to 40 µm. Used for positives only (they carry clear text) and for the
directional adversary (each tile rotated 90°).

**B3 — injected text.** Synthetic Greek text (rendered capitals, letter height 2.0 mm, stroke width
0.3 mm, line spacing equal to the median measured in step 1 below) added to B1 wrong-direction maps at
amplitudes a ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.4} (probability units, added inside glyphs, clipped to [0,1]).

**B4 — search fields.** The triage fields on PHerc1203 and PHerc0846A in `data/triage_*.csv`.

## Procedure

1. **Line period from labels only.** In each B2 segment's supervised area, take row profiles of the label map
   in 10 mm wide strips, find the dominant autocorrelation peak between 2.5 and 10 mm. The scoring band is
   [P10, P90] of these spacings across segments. This uses no score results.
2. **Score v2.** Tiles of 20 mm, step 10 mm, kept when coverage ≥ 0.6 and mask peak ≤ 1.0 (guards
   unchanged from v1). Per tile:
   - `peak` (v1): log of the vertical band-to-flank power ratio over the horizontal one, band from step 1;
   - `z` (v2): excess of vertical band power over a power law fitted to the tile's radially averaged
     spectrum outside the band, in units of the fit's residual standard deviation.
   Primary score: `z`. Map-level score: median over kept tiles.
3. **Baselines** on the same tiles/maps: separation (villa #1867: mean p above 0.5 minus mean p at or below
   0.5), mean p, fraction p > 0.5, anisotropy (v1 `score`).
4. **Calibration.** Leave-one-scroll-out: the tile threshold for a held-out scroll is the 95th percentile of
   the wrong-direction B1 tiles of the *other* scrolls.

## Primary outcomes and targets

| # | Outcome | Target |
|---|---|---|
| P1 | B1 direction choice, map-level primary score, over non-ambiguous segments | ≥ 90% correct, and ≥ the separation baseline |
| P2 | B1 tile-level AUC, text-direction vs wrong-direction tiles, pooled, leave-one-scroll-out | ≥ 0.80 |
| P3 | B2 tile-level AUC, published tiles vs their 90°-rotated copies | ≥ 0.90 |
| P4 | False-positive rate on held-out wrong-direction B1 tiles at the calibrated threshold | ≤ 5% (by construction on training scrolls; measured on held-out) |
| P5 | B3 amplitude for 50% detection at the calibrated threshold, with the real text contrast of B1 text-direction maps (mean p in labelled ink minus mean p in supervised non-ink) for comparison | reported |
| P6 | B4: fields passing the calibrated threshold after Benjamini–Hochberg at q = 0.10 | reported |

Secondary: the same outcomes with `peak` instead of `z`; per-scroll breakdown; the checkpoint-ranking pilot
(`scripts/ckpt_rank.py`, started before this document; its fixed analysis is in that file's docstring).

A missed target is reported with the measured value. No outcome is dropped.

## Deviations

(none yet)
