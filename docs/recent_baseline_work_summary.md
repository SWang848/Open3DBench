# Recent Baseline Work Summary

## 1. Motivation

This baseline was tested to understand whether a large candidate-solution space can be explored more efficiently by first reasoning over coarse regions and then selecting individual solutions within promising regions. The motivation is that full PPA evaluation is expensive, while many candidate solutions may share local structure. If regions carry useful signal about solution quality, a region-first strategy could reduce evaluation cost while still identifying competitive candidates.

## 2. Baseline Idea

The baseline uses generated candidate solutions for the `bp_multi` design family, represented by graph-diffused solution features and evaluated against precomputed metrics. Candidate solutions are grouped into balanced regions in feature space. Each region is summarized using simple aggregate information, including proxy quality statistics and low-dimensional feature statistics.

At a high level, the baseline tests a two-level DOPP procedure:

- Select informative regions using region-level D-optimal design.
- Evaluate representative solutions inside those regions.
- Train a simple surrogate from evaluated regions.
- Use the surrogate to select a second set of regions.
- Within selected regions, use a similar D-opt-plus-prediction procedure to select candidate solutions for evaluation.

The intent is to test whether structure at the region level can guide the evaluation budget better than treating all candidate solutions as one undifferentiated pool.

## 3. Main Observations

- The balanced clustering step produced approximately equal-sized regions. In the saved `bp_multi_3D_4` analysis, 100 regions contained about 100 to 102 candidates each.
- Good candidates were distributed across multiple regions rather than concentrated in a single area. The global top 10 candidates were spread across 9 regions, and the top 20 were spread across 17 regions.
- Region-level selection showed some useful signal. In the available region analysis, the selected regions after both rounds contained 5 of the global top 10 candidates, 9 of the top 20, 21 of the top 50, and 37 of the top 100.
- However, selecting a promising region did not guarantee that the inner solution selection found the best candidate in that region. In the analyzed run, inner D-opt found the true region minimum in only 15% of evaluated regions, with a mean oracle gap of about 0.014.
- Saved two-level runs found candidates near, but not equal to, the global best in the available metrics. The global best fitness in the checked `bp_multi_4` metrics was about 1.0679, while saved two-level runs reported best fitness values around 1.0751 and 1.0764.
- Candidate-level top-K coverage was limited in the saved results. One run with 212 oracle calls captured 1 of the top 10 and 2 of the top 20 evaluated candidates. A later first-pass run with 583 oracle calls captured 1 of the top 10 and 1 of the top 20.
- TODO: Confirm the latest post-edit run results, because the current implementation has evolved beyond some saved result bundles and notes.

## 4. What This Baseline Shows

This baseline provides early evidence that region-level structure is meaningful: high-quality solutions are not uniformly random with respect to the constructed regions, and selected regions can contain a nontrivial fraction of top-ranked candidates. This supports the idea that a coarse-to-fine selection strategy is worth studying.

At the same time, the baseline does not yet show that the current two-level procedure reliably recovers the best candidate solutions under a small evaluation budget. The main bottleneck appears to be the handoff from promising regions to specific candidate choices within those regions. The results suggest that region selection and solution selection should be analyzed separately: the former shows signal, while the latter still loses many strong candidates.

As an early baseline, this is useful because it separates two questions: whether regions are informative, and whether the current features and surrogate/D-opt procedure are sufficient to exploit those regions.

## 5. Current Limitations

- The current region definition may be too naive or too dependent on the chosen feature representation.
- Region-level summaries may not capture enough information to distinguish regions with truly strong candidates from regions with only average quality.
- The best solution in a region may not reflect the overall quality or predictability of that region.
- The inner solution-selection step can miss the true best candidate even when the selected region is promising.
- The observed results come from limited saved runs, so stability across random seeds, designs, and region counts is still uncertain.
- The current analysis does not fully explain why some selected regions contain top candidates while others do not.
- TODO: Add a controlled comparison against single-stage D-opt and simpler random or proxy-based baselines using the same oracle budget.

## 7. Short Summary for Research Plan

We explored a two-level DOPP baseline for budget-aware candidate selection in a large 3D placement candidate space. The baseline was designed to test whether candidate solutions can first be organized into meaningful regions and then searched more selectively within promising regions. Candidate solutions were represented using graph-diffused features, grouped into balanced regions, and summarized with simple region-level statistics. The method then used D-optimal design and lightweight surrogate prediction at two levels: first to choose regions for evaluation, and then to choose candidate solutions within selected regions.

The main takeaway is that the region abstraction appears to contain useful signal, but the full baseline is not yet strong enough to reliably recover the best candidate solutions. In the available `bp_multi` analysis, high-quality candidates were distributed across many regions, and the selected regions after two rounds contained several globally top-ranked candidates. This suggests that the grouping is not arbitrary and that region-level search may be a useful way to allocate evaluation budget. However, the evaluated candidate coverage remained limited: saved runs captured only a small number of the global top candidates, and the inner solution-selection stage often missed the true best solution inside selected regions.

These results support the baseline as a useful early diagnostic rather than a final method. It provides evidence that region-level structure can help narrow the search space, but it also shows that the current region summaries, feature representation, and within-region selection strategy are insufficient to fully exploit that structure. The baseline also helps separate region-selection quality from within-region candidate-selection quality. The next stage should therefore focus on understanding what makes a region predictive of strong PPA outcomes, improving the features or statistics used to represent regions, and evaluating the method under controlled comparisons against simpler baselines. TODO: Re-run the finalized version of the current implementation and report stable results across multiple designs or seeds.
