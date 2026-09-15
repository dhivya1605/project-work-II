# S4 — Joint Constraint and Boundary-Aware Synthetic Data Generation
## Final Methodology Report

> **IMPORTANT LIMITATION STATEMENT**
> The source dataset (`crop-dataset.xlsx`) is a **range/specification-based dataset** — it contains
> minimum and maximum bounds for approximately 49 crop classes, NOT thousands of observational measurements.
> S4 does NOT claim to have learned the true real-world distribution.
> S4 does NOT claim to have discovered real correlations unless the source specification supports them.
> All dependency estimates are derived from 49 crop midpoint values (one per crop) — a conservative estimate.
> S4 does NOT train any machine-learning model during generation.

---

## A. Methodology

### A.1 What the Three Baseline Frameworks Do

| Framework | Core Strategy | Dependency | Selection |
|-----------|--------------|-----------|-----------|
| **Baseline-1** (10_Synthetic_Crop_Data_AHAPSF1.py) | Joint 8D Sobol → full 8-feature Gaussian copula with HARD-CODED crop-group correlation matrices | Full 8-feature copula; 5 hard-coded crop groups | First N valid rows after 1.2× oversampling |
| **Baseline-2** (7_Synthetic_Crop_Data_AHAPSF.py) | Feature-by-feature independent: Sobol/LHS/TruncNorm/Beta/Copula | NPK only; HARD-CODED corr=0.5 | Direct 600 — no selection, no deduplication |
| **Baseline-3** (AHAPSF2/3_framework.py) | Adaptive per-feature sampler selection (data-driven algorithm choice) | NPK Spearman copula; other features independent | Jitter on duplicates only; no maximin selection |

### A.2 Main Limitations of the Baseline Frameworks

1. **Feature-wise concatenation as main strategy**: All three generate features independently
   (or at most group NPK via copula) and concatenate. No framework selects final samples
   based on joint 8D vector quality.

2. **No 8D maximin diversity selection**: Baseline-1 selects the first N valid rows.
   Baseline-2 generates exactly N with no selection. Baseline-3 jitters duplicates but
   does not optimize joint-space diversity.

3. **Fabricated or hardcoded correlations**: Baseline-1 uses hard-coded crop-group correlation
   matrices (cereals/pulses/oilseeds/vegetables/commercial) with fixed values not derived from data.
   Baseline-2 hard-codes corr=0.5 for NPK for all crops regardless of actual structure.

4. **Fixed distribution parameters**: Baseline-2 uses Beta(5,5) for all crops regardless of
   skewness. Baseline-1 uses fixed copula regardless of feature independence.

### A.3 What S4 Changes

S4 operates on COMPLETE 8-dimensional feature vectors, not individual features:

1. **Joint 8D candidate generation**: Four mechanisms — Sobol-8D, LHS-8D, Gaussian Copula-8D,
   and Adaptive Hybrid-8D — all generate complete 8D vectors (not individual features).

2. **Conservative data-driven dependency**: Spearman correlations are computed across all
   49 crop midpoints. Only correlations with |rho| >= 0.25 are preserved.
   This prevents fabricating correlations not supported by source data.

3. **Joint feasibility score**: Each complete 8D candidate vector is scored holistically:
   boundary validity + interior distance + dependency consistency + marginal plausibility.

4. **Greedy maximin diversity selection**: 600 samples are selected from 1,800 candidates
   by maximizing minimum pairwise distance in normalized 8D space, weighted by feasibility.
   This is fundamentally different from all three baselines.

5. **Range-aware normalization**: All joint-space operations use normalized [0,1]^8 space,
   preventing large-scale features (WATERREQUIRED, N) from dominating distance calculations.

### A.4 Why Joint Feature-Space Generation May Help Downstream Classification

When features are generated independently and concatenated, no mechanism prevents unlikely
feature combinations (e.g., very high temperature with very high relative humidity AND very
low water requirement simultaneously). Joint generation with feasibility scoring reduces
such combinations.

The greedy maximin selection ensures maximum spread across the 8D feature space per crop,
which provides denser, more informative training signal for downstream ML classifiers
operating on all 8 features simultaneously.

---

## B. Novelty

The novelty of S4 is NOT merely combining existing algorithms. The novel contribution is:

1. **Complete 8D vector evaluation**: The joint feasibility score treats each candidate as
   a whole 8D entity, not independent scalar values.

2. **Maximin × Feasibility joint selection criterion**: No baseline uses this. The selection
   criterion simultaneously maximizes spatial diversity AND respects distribution quality.

3. **Conservative thresholded dependency**: Instead of hard-coding correlations (BL-1) or
   assuming independence (BL-2, BL-3 for non-NPK), S4 uses a principled threshold on
   data-derived Spearman correlations.

4. **4-mechanism joint candidate pool**: Operating all 4 mechanisms in full 8D normalized
   space (vs. feature-by-feature generation) creates a richer, more diverse candidate pool
   from which the best 600 can be selected.

---

## C. Quality Results

| Metric | Value |
|--------|-------|
| A. Boundary Validity (OOB Rate) | 0.0% |
| A. Negative Values | 0 |
| B. Avg Range Coverage | 0.9995 |
| C. Avg Wasserstein Distance | 6.8458 |
| D. Avg Joint Dispersion | 0.6747 |
| E. Correlation Frobenius Error | 0.6284 |
| F. Normalized 8D Joint Coverage | 0.0091 |
| G. Avg Maximin Distance | 0.28345 |
| H. Exact Duplicate Rate | 0.0% |
| I. Near-Duplicate Rate | 0.0% |
| J. Outlier Rate | 0.0408% |
| K. Cross-Crop Overlap | 0.2493 |
| L. Avg Per-Crop Coverage | 0.9995 |
| **Synthetic Quality Score (SQS)** | **86.62 / 100** |

---

## D. Dependency Analysis Summary

- Dependency threshold used: |Spearman rho| >= 0.25
- Source: 28 unique feature pairs across 49 crop specification midpoints
- Retained dependencies: 14 / 28
- Features with dependencies preserved: see S4_dependency_analysis.xlsx

**Note**: With only 49 specification rows (one per crop), correlation estimates have
wide confidence intervals (approximately ±0.28 for n=49 at p=0.05). Conservative
thresholding at 0.25 is appropriate. The S4 framework does NOT claim these represent
true agronomic correlations — only that they are consistent with the available specification.

---

## E. Generation Summary

- Total crops processed: 49
- Total synthetic samples: 29400
- Samples per crop: 600
- Oversampling factor: 3× (1800 candidates per crop)
- Selection method: Greedy Maximin in normalized 8D space × Joint Feasibility Score
- Total execution time: 44.3 seconds

---

## F. Limitations

1. **Range-only source**: The source dataset provides only min/max bounds per crop, not
   observational measurements. S4 cannot recover information that doesn't exist in the source.

2. **49-point correlation estimates**: With only 49 crops, Spearman correlation estimates
   for 28 feature pairs are imprecise. Conservative thresholding mitigates fabrication risk.

3. **No causal model**: S4 does not model agronomic causal relationships — only statistical
   patterns observable in the specification table.

4. **Integer rounding**: Rounding to integers (TEMP, CROPDURATION, etc.) reduces diversity,
   especially for crops with narrow ranges. Post-rounding jitter is applied conservatively.

---

*Report generated automatically by S4_framework.py. Total execution time: 44.3s*
