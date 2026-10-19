# Results

Raw outputs behind every number in the write-up and README. Regenerate with `make benchmark`
(writes `data/bench/results.json`, copied here) and `scripts/benchmark/ckpt_rank.py`.

| File | What it holds |
|---|---|
| `bench_results.json` | Pre-registered benchmark B1/B3/B4 (see `PREREG.md`). Keys below. |
| `bench_p3.json` | Outcome P3: published 2.4 µm predictions vs their 90°-rotated copies, tile AUC (`all_tiles`, `n`) and the same on tiles where labels confirm ink (`label_confirmed`, `n_label`). |
| `bench_exploratory_label_confirmed.json` | Exploratory (not pre-registered): v3 on the B1 text-direction tiles where labels confirm ink (`auc_label`, `n_label`) and on those without labelled ink (`auc_unlabelled`), against all wrong-direction tiles (`n_neg`); `tpr_label_at_5pct` is the share flagged at a 5% false-positive rate. |
| `ckrank_scores.json` | Checkpoint pilot, one row per checkpoint × depth direction on PHerc0841 w00: label `auc`, text scores `text` (v1) and `text_v3` (exploratory), `separation`, number of scored `tiles`. |
| `ckrank_scores_fixed_analysis.json` | The same rows as first scored, before the exploratory v3 column was added. |
| `ckrank_summary.txt` | Spearman correlations, picked checkpoints and depth-direction accuracy from the pilot. |

## Keys in `bench_results.json`

| Key | Meaning |
|---|---|
| `band_mm`, `spacing_mm` | Line band (P10–P90) and median line spacing measured from the labels |
| `b1_segments` | Per segment: label AUC both directions, label shift used, text direction, `ambiguous`, real ink contrast, map-level scores |
| `P1_direction` | Depth direction chosen correctly, per score (`correct` of `n`) |
| `P2_tile_auc` | Tile AUC, text-direction vs wrong-direction tiles, pooled and `per_scroll` |
| `P4_loso` | Leave-one-scroll-out false- and true-positive rates at the 95th-percentile threshold |
| `P5_injection` | Detection rate of synthetic Greek text added at each amplitude, at the calibrated threshold |
| `P6_search` | Clean search fields tested with Benjamini–Hochberg at q = 0.10 |

Score names: `s3` = v3 row-profile score (primary), `z` = v2, `peak` = v1, `aniso` = anisotropy, `mean` = mean ink,
`frac` = fraction above 0.5, `sep` / `sep_map` = separation rule from villa #1867.
