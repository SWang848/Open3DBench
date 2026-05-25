# Two-Level DOPP: Conceptual Description (Current Version)

## Purpose
Two-Level DOPP is a budget-aware candidate selection strategy designed to reduce expensive full-solution evaluations. Instead of evaluating all candidate solutions directly, it first selects promising **regions** of the solution space, then selects representative **solutions** inside those regions.

In short:
- Region level decides **where to spend budget**.
- Solution level decides **which specific candidates to evaluate**.

This creates a structured exploration process that balances broad coverage and focused exploitation.

## Inputs and Data Objects
The current algorithm uses three aligned solution-level data views:
- **Solution feature matrix (`X`)**: numeric representation of each candidate solution.
- **Fitness vector (`y`)**: target quality value (lower is better in current usage).
- **Candidate keys**: stable IDs used to keep rows aligned across files and selections.

It also uses two proxy metrics per candidate:
- **Cut size**
- **Area imbalance**

These proxies are used only for region-level feature construction.

## Core Idea of the Two Levels
1. Cluster all solutions into balanced regions in feature space.
2. Build one feature vector per region.
3. Run D-optimal design on region features to pick informative regions (Round 1).
4. Inside each picked region, run D-optimal design again on solution features to pick informative solutions.
5. Use Round-1 region outcomes to train a region-level surrogate model.
6. Use the surrogate to pick additional unlabeled regions (Round 2).
7. Repeat inner solution selection and aggregate final evaluated candidates.

## Solution-Level Features
The algorithm uses a dense solution feature matrix `X` loaded from the feature bundle (`.npy` object with `features` and `candidate_keys`).

Detailed definition:
- **One row = one candidate solution** (matched by `candidate_keys`).
- **One column = one numeric descriptor** of that candidate.
- The matrix is aligned with fitness labels (`y`) by key, so feature rows, keys, and fitness targets stay in the same order.
- The workflow treats these as pure numeric vectors (no semantic feature-name dependency during optimization).

Current solution feature construction is graph-diffused partition features:
- Base per-node feature names:
  - `partition_label`
  - `incident_cut_net_count`
  - `flip_area_imbalance_gain`
  - `cross_tire_cell_conectivity`
  - `hierarchy_cohesion`
- Diffusion levels (default): hop-0 (`X_p`), hop-1 (`S X_p`), hop-2 (`S^2 X_p`)
- Per solution, all nodes × all hops × all base feature channels are concatenated and flattened into one vector.
- If solution-feature PCA is enabled during feature-bundle construction, this flattened vector is compressed (feature type `graph_diffused_pca`); otherwise it remains `graph_diffused`.

How they are used:
- **Standardized solution features (`X_std`)** are used for balanced clustering and PCA projection used by region-level aggregation.
- **Original solution features (`X`)** are used for inner (within-region) D-optimal selection, so candidate picking is done in the original feature space.

## Region-Level Features
Each region is converted into one feature vector by aggregating member solutions.

Current region feature groups are:
- Cut size statistics: min, max, mean, std
- Area imbalance statistics: min, max, mean, std
- Region size (number of member solutions)
- PCA-embedding statistics over region members (**PCA components fixed to 10**):
  - per-dimension mean
  - per-dimension std

So each region feature vector is:
- 9 base dimensions from proxy stats + region size
- plus `2 × 10` PCA summary dimensions

After construction, QR-based rank cleanup removes linearly dependent region feature columns before region-level D-optimal design.

## Balanced Region Formation
Solutions are clustered into `K` regions after standardization.

Balancing mode:
- **Constrained balancing** (constrained k-means) is used.

Goal: keep region sizes within a controlled range so the region-level process does not over-focus on very large clusters or under-represent smaller ones.

## Round 1 (Region-First Selection)
Round 1 runs D-optimal design on region features to get region weights. Top weighted regions are selected for expensive evaluation.

For each selected region:
- Run inner solution-level D-optimal design on that region’s member solutions.
- Select a fraction of solutions by inner D-optimal weights.
- Evaluate selected solutions (via current precomputed fitness lookup flow).
- Record the best fitness found per region.

This produces labeled region outcomes used for surrogate training.

## Region Surrogate
A region-level regression model is fit to map region features to observed region quality (best fitness in that region from Round 1).

Current behavior:
- Uses **LinearRegression**.

The surrogate is used to score unlabeled regions for Round 2.

## Round 2 (Surrogate-Guided Selection)
Among regions not selected in Round 1:
- Predict region quality with the surrogate.
- Pick regions with best predicted quality (lowest predicted fitness).
- For those regions, run the same inner solution-level D-optimal selection and evaluation process.

This adds a second exploration wave guided by learned structure rather than only first-pass D-optimal weights.

## Final Outputs and Metrics
The pipeline aggregates both rounds and reports:
- All evaluated solution indices/keys
- Best solution found and its fitness
- Oracle call count (number of unique evaluated solutions)
- Per-round selected regions and per-region best outcomes
- Surrogate training metrics
- Round-2 prediction-vs-truth agreement (Kendall tau on eligible regions)
- Top-K coverage against the global true ranking

## Practical Interpretation
This implementation can be viewed as:
- **Phase A (representation and partitioning):** standardize, cluster, aggregate regions.
- **Phase B (structured sampling):** D-opt at region level, then D-opt within selected regions.
- **Phase C (adaptive refinement):** learn from Round 1 region outcomes and pick better Round 2 regions.

Compared with single-stage D-opt on all solutions, this two-level version emphasizes:
- Better budget allocation across diverse parts of the search space
- Improved sample efficiency under expensive evaluation constraints
- A clean handoff between unsupervised partitioning and supervised refinement

