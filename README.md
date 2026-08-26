# Forensics Project (CASIA 2.0) — Autosufficient Build

**Start here**: `results/phase5-report/forensics_final.ipynb`

This directory contains a complete, self-contained snapshot of the forensics detector pipeline and all analyses. No external paths required.

## Quick Facts

- **Three key findings**: 
  1. **Q1: No calibration benefit.** Pipeline-specific calibration does not improve detection performance (|ΔMCC| ≤ 0.0044 at matched operating point). *Negative result, retraction of prior finding in §11.2.*
  2. **Donor memorization.** Frozen weights generalize poorly to unseen donors (LR+ degrades 11.83 → 5.37). *Discussed in notebook §9.*
  3. **Codec comparison inconclusive.** At 0.5166 bpp, no pipeline provably preserves forensic signal better than others. Ranking unstable across robustness checks; median ΔF1 ≈ 0. *Q2 result, §19–21.*

- **Three retractions**:
  1. JPEG-AI compression benefit (§11.2, artifact of single-filter fusion)
  2. Optimistic MCC estimates from overfit calibration (§6, fixed via 3-fold CV)
  3. All-positive F1 baseline claim of improvement (detector still 6.1pp below banality check)

- **Detector status**: Baseline warning remains active (F1 0.19–0.23 vs. 0.258 all-positive). Codec ranking is *relative* (which degrades more), not absolute preservation claim.

- **All numbers in `results/phase5-report/` and learned-detector branches are IN-DOMAIN.**

## Structure

```
.
├── README.md                    ← you are here
├── CLAUDE.md                    ← scientific rules & protocol invariants
├── EXPERIMENT_LOG.md            ← chronological record of runs & decisions
├── src/forensics/
│   ├── detector.py              ← per-window score maps
│   ├── calibration.py           ← threshold search & CV selection
│   ├── metrics.py               ← confusion matrix → MCC/F1/LR+
│   ├── stats.py                 ← bootstrap, CI, randomization tests
│   ├── compression.py           ← codec wrappers (JPEG, OSN, JPEG-AI)
│   ├── preservation.py          ← separability measurement
│   ├── dataset.py               ← CASIA2 split & split guards
│   └── params.py                ← configuration (detector, rate, target FPR)
├── notebooks/
│   ├── forensics_final.py       ← source → notebooks/build_notebook.py generates .ipynb
│   ├── build_notebook.py        ← Jupyter notebook builder (do not --run; it moves results/)
│   └── make_failure_panels.py   ← qualitative failure examples
├── tests/
│   └── test_forensics.py        ← 57 guards: split logic, metrics, bootstrap, resolvers
├── results/
│   ├── phase5-report/           ← final report
│   │   ├── forensics_final.ipynb    (959 KB)
│   │   ├── conclusion.txt           (3.7 KB)
│   │   └── failure_panels.npz       (1.8 MB, qualitative examples)
│   ├── 20260818-141123/         ← **frozen classic detector** (§19 final state)
│   ├── 20260818-171351/         ← deep-learning branch (§21)
│   ├── 20260817-193239/         ← iteration 16 (macro objective)
│   ├── [... 8 more runs, all cited in EXPERIMENT_LOG.md ...]
│   └── archive_calibration_question/  ← archived Q1-only results (§11, pre-retraction)
└── archive/                     ← (currently empty; old monolithic pipeline)
```

## Regenerate

### Notebook from source (no training):
```bash
cd notebooks
python build_notebook.py
# Output: Markdown cell tree in stdout (for debugging)
# Side effect: builds .ipynb from forensics_final.py
# Result: forensics_final.ipynb in working directory
```

**IMPORTANT:** Do NOT run `build_notebook.py --run`; it would move `results/latest` and break resolver §5.11.

### Run tests (all splits and metrics):
```bash
pytest tests/test_forensics.py -v
# 57 assertions covering split integrity, bootstrap logic, metrics aggregation, path resolution
```

### Conclusion text:
```bash
python -c "from results.phase5_report import generate_summary; print(generate_summary())"
# (Or read results/phase5-report/conclusion.txt directly)
```

## Data Integrity Checks (Built-in)

Every notebook cell asserts invariants from CLAUDE.md §5:
- **Split at image level**: Train / Validation / Test / Calibration never mix windows from the same source image.
- **Rate per source pixel**: bpp calculated on original image dimensions, never resampled.
- **One frozen threshold**: all codec arms at same operating point.
- **MCC on pooled confusion matrix**: not mean-of-per-image metrics (eliminates bias from small counts).
- **No silent defaults**: missing artifacts raise loudly; resolver raises iff artefact does not exist.
- `results/latest` symlink (if present) points to the frozen classic run.

If any cell raises `AssertionError`, the protocol invariant has been violated; fix before trusting results.

## External Dependencies

- **Code**: Python 3.10+, PyTorch, OpenCV, NumPy/Pandas, Jupyter
- **Codecs**: JPEG (pillow), libopenslide (OSN), JPEG-AI (vendored, ~1-2 min/image)
- **Data**: CASIA2 Tampering Detection dataset (500 images, ~1.2 GB raw; must be at `CASIA2/` relative to working dir)

No cloud APIs, no internet required after initial clone.

## Metrics Glossary

- **MCC** (Matthews Correlation Coefficient): balanced confusion-matrix metric, range [−1, 1].
- **F1**: harmonic mean of precision & recall; all-positive baseline F1 = 0.258 (median on test set).
- **LR+** (positive likelihood ratio): sensitivity / (1 − specificity); >2 is weak, >10 is strong.
- **Separability** (abs_separability in code): |median Sobel gradient on tampered − on authentic|, detec­tor-agnostic measure of forgery strength.
- **ΔF1, ΔMCC**: change in metric under codec compression vs. original image.
- **bpp** (bits per pixel): rate, calculated on source image dimensions.

## Mapping: Old → New Paths

(To be filled in §32 of EXPERIMENT_LOG.md)

---

**Questions?** See EXPERIMENT_LOG.md for the full chronology of discoveries, retractions, and decisions. All open problems and known limitations are listed at the end of the relevant iteration entries.
