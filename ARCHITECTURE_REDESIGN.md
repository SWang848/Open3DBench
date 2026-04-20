# Clean Architecture Redesign

## Goal
Reorganize Open3DBench so the main algorithm owns workflow, contracts, and evaluation policy, while `Place-3D` and `OpenROAD` become isolated engines behind adapters. The redesign favors clarity and extensibility over backward compatibility.

## Current Problems
- `Place-3D/dreamplace/Placer_3D.py` combines algorithm selection, placement orchestration, DEF rewriting, logging, and external tool execution.
- `OpenROAD-3D/flow/scripts_3D/autoflow.sh` reaches directly into `Place-3D/install/results/...` and assumes repo-relative artifacts.
- `DOPP.sh` is the real system orchestrator today, so architecture lives in shell scripts instead of Python modules.
- `HMSA_solution_eval/get_metrics.py` parses OpenROAD-specific log filenames and scratch layouts, so evaluation logic cannot be reused cleanly.
- `Place-3D/dreamplace/Regression.py` imports `cal_fitness_score` from `HMSA_solution_eval/get_metrics.py`, which creates a cross-layer dependency from learning code into simulator-specific evaluation code.
- `graph_construction()` exists in both `Place-3D/dreamplace/Partitioner.py` and `Place-3D/dreamplace/HierarchyMultiObjectiveSA.py`, which invites drift.

## Design Principles
- Keep the core simulator-agnostic.
- Make artifacts explicit and typed.
- Treat shell, Slurm, and containers as runtime concerns, not architecture boundaries.
- Separate algorithm logic from engine invocation.
- Parse engine-specific results inside adapters, then normalize once in evaluation.
- Preserve numerical kernels until boundaries are stable; move orchestration first.

## Target Repo Layout
```text
Open3DBench/
  core/
    domain/
      benchmark_case.py
      candidate.py
      partition_plan.py
      placement_artifact.py
      evaluation_result.py
    interfaces/
      partition_strategy.py
      placement_engine.py
      evaluation_engine.py
      artifact_store.py
      workflow_runtime.py
    orchestration/
      experiment_runner.py
      candidate_pipeline.py
      batch_evaluator.py
  algorithms/
    partition/
      mincut_partitioner.py
      graph_builder.py
    search/
      hmsa_search.py
      tpgnn_search.py
      dopt_selector.py
    learning/
      feature_builder.py
      regression_model.py
  adapters/
    place3d/
      engine.py
      artifact_mapper.py
      def_postprocess.py
      timer_bridge.py
    openroad/
      engine.py
      log_parser.py
      metrics_parser.py
      hotspot_parser.py
    filesystem/
      local_artifact_store.py
    runtime/
      slurm_runtime.py
      apptainer_runtime.py
      local_runtime.py
  evaluation/
    metrics/
      normalizer.py
      fitness.py
      pareto.py
    reports/
      metrics_report.py
      plots.py
  apps/
    run_experiment.py
    run_candidate_search.py
    run_evaluation.py
  legacy/
    place3d/
    openroad/
```

## Core Contracts
The core should define the only contracts that algorithms and adapters are allowed to share.

### Domain Objects
- `BenchmarkCase`: design identity, benchmark inputs, and technology metadata.
- `CandidateKey`: stable identifier for one partition or placement candidate.
- `PartitionPlan`: upper/lower die assignment plus algorithm-side cost metadata.
- `PlacementRequest`: benchmark, optional seed, partition plan, and engine parameters.
- `PlacementArtifact`: normalized outputs from a placement engine such as DEF paths, logs, and placement summaries.
- `EvaluationRequest`: placement artifact plus requested metrics and evaluation configuration.
- `EvaluationResult`: normalized PPA, timing, thermal, congestion, and provenance metadata.
- `ExperimentRecord`: end-to-end run manifest tying together candidate generation, placement, and evaluation.

### Interface Sketch
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

@dataclass
class BenchmarkCase:
    name: str
    config_path: Path
    technology: str

@dataclass
class PartitionPlan:
    candidate_key: str
    upper_die_nodes: list[str]
    lower_die_nodes: list[str]
    cut_size: float | None = None
    area_imbalance: float | None = None

@dataclass
class PlacementArtifact:
    candidate_key: str
    engine: str
    top_def: Path
    metadata: dict

@dataclass
class EvaluationResult:
    candidate_key: str
    engine: str
    metrics: dict[str, float | None]
    artifacts: dict[str, Path]

class PartitionStrategy(Protocol):
    def generate(self, case: BenchmarkCase) -> list[PartitionPlan]: ...

class PlacementEngine(Protocol):
    def place(self, case: BenchmarkCase, plan: PartitionPlan) -> PlacementArtifact: ...

class EvaluationEngine(Protocol):
    def evaluate(self, case: BenchmarkCase, artifact: PlacementArtifact) -> EvaluationResult: ...
```

## Layer Responsibilities

### `core/`
- Owns workflow state, contracts, manifests, and orchestration rules.
- Knows which steps happen, but not how `Place-3D` or `OpenROAD` implement them.
- Replaces shell-driven coupling from `DOPP.sh`.

### `algorithms/`
- Contains candidate generation and model logic only.
- Produces `PartitionPlan` or candidate scores, never repo-relative file paths.
- Owns one canonical `graph_builder.py` to replace duplicated `graph_construction()`.

### `adapters/`
- Encapsulate all engine-specific formats, commands, and log parsing.
- Convert core contracts into tool-specific inputs and convert results back into normalized artifacts.
- Own runtime launch details for local, Slurm, and container execution.

### `evaluation/`
- Calculates normalized fitness, Pareto ranking, plots, and benchmark reports.
- Accepts normalized metrics dictionaries, not raw OpenROAD log trees.
- Becomes the single home for logic like `cal_fitness_score()`.

## Data Flow
```mermaid
flowchart LR
  benchmarkCase[BenchmarkCase] --> partitionStrategy[PartitionStrategy]
  partitionStrategy --> partitionPlan[PartitionPlan]
  partitionPlan --> placementEngine[PlacementEngineAdapter]
  placementEngine --> placementArtifact[PlacementArtifact]
  placementArtifact --> evaluationEngine[EvaluationEngineAdapter]
  evaluationEngine --> evaluationResult[EvaluationResult]
  evaluationResult --> metricsLayer[EvaluationMetrics]
  metricsLayer --> experimentRecord[ExperimentRecord]
```

## How Current Code Maps To The New Boundaries

### Move into `core/`
- Extract orchestration intent from `DOPP.sh` into `core/orchestration/experiment_runner.py`.
- Replace implicit scratch directory contracts from `architecture.txt` with explicit artifact-store models.
- Move step ordering, dependency rules, and run manifests out of shell into Python orchestration.

### Move into `algorithms/`
- `Place-3D/dreamplace/Partitioner.py` becomes `algorithms/partition/mincut_partitioner.py`.
- `Place-3D/dreamplace/HierarchyMultiObjectiveSA.py` becomes `algorithms/search/hmsa_search.py`.
- `Place-3D/dreamplace/TPGNN.py` becomes `algorithms/search/tpgnn_search.py`.
- `Place-3D/dreamplace/FeatureConstructionByManual.py`, `FeatureConstructionByNN.py`, `FeatureTraining.py`, `D-opt.py`, and the model-facing parts of `Regression.py` become `algorithms/learning/...`.
- Shared graph construction moves into one `algorithms/partition/graph_builder.py`.

### Keep behind `adapters/place3d/`
- `Place-3D/dreamplace/Placer_3D.py` and `Placer_3D_hmsa.py` should be split so engine invocation and DEF post-processing stay in the adapter while search policy moves out.
- `Place-3D/dreamplace/PlaceDB.py`, `NonLinearPlace.py`, `BasicPlace.py`, `PlaceObj.py`, `Timer.py`, and `ops/` stay engine-side because they are specific to the placement backend.
- `Place-3D/run_3D.sh` becomes a thin runtime entrypoint that calls `apps/run_experiment.py`.

### Keep behind `adapters/openroad/`
- `OpenROAD-3D/flow/scripts_3D/autoflow.sh` becomes adapter-internal runtime glue.
- `OpenROAD-3D/flow/autoflow_hmsa_cc.slurm` becomes a runtime backend implementation, not a pipeline definition.
- `OpenROAD-3D/flow/scripts_3D/get_metrics.py` should be split into parser helpers inside `adapters/openroad/`.
- HotSpot outputs should be parsed in the same adapter layer and emitted as normalized metrics.

### Move into `evaluation/`
- `HMSA_solution_eval/get_metrics.py` should be split into:
  - adapter-level raw report collection for OpenROAD file names
  - evaluation-level metric normalization, fitness scoring, Pareto plotting, and CSV generation
- `Regression.py` should depend on `evaluation/metrics/fitness.py`, not on `HMSA_solution_eval/get_metrics.py`.

## Boundary Rules
- `core/` cannot import `Place-3D`, `OpenROAD-3D`, or any shell scripts.
- `algorithms/` can depend on `core/domain` and `core/interfaces`, but not on adapter internals.
- `evaluation/` can depend on normalized `EvaluationResult`, but not on raw simulator log filenames.
- Adapters may depend on legacy directories and runtime tools, but only adapters can do so.
- Only runtime adapters may call Slurm, Apptainer, `os.system`, or subprocess commands.

## High-Risk Refactor Areas

### 1. `Placer_3D.py` is a monolith
It currently mixes several concerns:
- benchmark loading
- partition-mode branching
- DEF rewriting
- external detailed placement execution
- filesystem side effects

This file should be split first, but carefully, because it is both an entrypoint and an engine wrapper.

### 2. Evaluation depends on raw OpenROAD filenames
`HMSA_solution_eval/get_metrics.py` looks for files such as `openroad_logs/6_report.json` and `5_2_route.json`. That contract should move into `adapters/openroad/log_parser.py`, while evaluation consumes a normalized metric object.

### 3. Cross-layer import in `Regression.py`
This import:
```python
sys.path.append(os.path.join(root_dir, "HMSA_solution_eval"))
from get_metrics import cal_fitness_score
```
must be replaced with a clean import from `evaluation/metrics/fitness.py`.

### 4. Artifact layout is implicit
The scratch layout in `architecture.txt` and the folder assumptions in `DOPP.sh` should become explicit through an `ArtifactStore` interface and manifest files.

### 5. Runtime concerns leak into architecture
Slurm arrays, dependency handling, and apptainer setup belong in `adapters/runtime/`, not in the algorithm pipeline definition.

## Recommended Migration Phases

### Phase 1: Introduce contracts without moving engines
- Add `core/`, `algorithms/`, `adapters/`, and `evaluation/`.
- Define the domain objects and protocol interfaces.
- Wrap the current `Place-3D` and `OpenROAD` entrypoints with thin adapters.
- Keep legacy scripts working through the adapters during this phase.

### Phase 2: Extract evaluation first
- Move `cal_fitness_score()`, Pareto plotting, and report generation into `evaluation/`.
- Leave raw OpenROAD parsing in an adapter-level parser.
- Update `Regression.py` to depend on the new evaluation module.

### Phase 3: Separate search from placement
- Move HMSA, TPGNN, D-opt, and feature builders into `algorithms/`.
- Replace duplicate graph construction with a shared graph builder.
- Make algorithm outputs return `PartitionPlan` and candidate metadata only.

### Phase 4: Shrink the `Place-3D` adapter surface
- Split `Placer_3D.py` into engine adapter code, DEF transforms, and runtime entrypoints.
- Keep DREAMPlace numerical kernels in place.
- Remove policy decisions from engine-facing modules.

### Phase 5: Move orchestration out of shell
- Replace `DOPP.sh` with `apps/run_experiment.py`.
- Move Slurm and apptainer behavior behind runtime adapters.
- Generate per-run manifests so results can be replayed without hard-coded paths.

### Phase 6: Retire legacy coupling
- Stop importing across top-level legacy folders.
- Stop depending on `Place-3D/install/results/...` and `HMSA_solution_eval/...` as architecture contracts.
- Keep `legacy/` only as compatibility shims until old scripts are removed.

## First Concrete Refactors To Do Next
1. Create `evaluation/metrics/fitness.py` and move `cal_fitness_score()` there.
2. Create `algorithms/partition/graph_builder.py` and use it from both partitioning and HMSA.
3. Introduce `PlacementEngine` and `EvaluationEngine` wrappers around the current `Placer_3D.py` and `autoflow.sh`.
4. Replace the top-level control flow in `DOPP.sh` with a Python `ExperimentRunner`.

## Success Criteria
- A new evaluation backend can be added without editing algorithm code.
- A new candidate-generation algorithm can be added without editing `Place-3D` or `OpenROAD` scripts.
- Metrics and fitness code run from normalized results rather than simulator-specific file trees.
- Runtime choices such as local execution, Slurm, or containerization are selected through adapters instead of changing business logic.
