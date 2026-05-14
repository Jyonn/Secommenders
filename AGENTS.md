# AGENTS

This repository hosts `Secommenders`, a benchmark for semantic-aware sequential recommendation.

## Mission

`Secommenders` studies how item semantic information should be incorporated into sequential recommenders.

The benchmark should answer questions such as:

- Which semantic integration mechanism works best in sequential recommendation?
- When does semantic input help beyond pure ID-based modeling?
- How should history representation and next-item target space be decoupled?
- Which methods help most for cold-start, tail items, sparse users, and metadata-rich settings?

## Scope

This project is not limited to comparing `ID`, `SID`, `text`, or `embedding` as item formats.

Its main goal is to benchmark semantic integration mechanisms, including:

- `ID-only` sequential recommenders
- `ID + semantic feature` recommenders
- `semantic-dominant` sequential recommenders
- `semantic target` recommenders such as `SID` or code prediction
- `retrieve-then-rank` pipelines that use semantic content

## Core Design Principles

### 1. Decouple Input and Target

History item representation and next-item prediction target must be treated as separate design axes.

Examples:

- History can be represented by `ID + text embedding`
- History can be represented by `semantic embedding`
- Target can still be `item ID`
- Target can still be `SID`

Do not assume input representation and output representation must be identical.

### 2. Use a Modular Benchmark Abstraction

Every method should be described with explicit modules whenever possible:

- `ItemSemanticEncoder`
- `HistorySequenceEncoder`
- `SemanticFusion`
- `TargetProjector`
- `TrainingObjective`
- `InferenceRetriever`

This abstraction is more important than matching original paper naming.

### 3. Evaluate Mechanism, Not Just Format

The benchmark should compare where and how semantics enter the system, including:

- input-level semantic injection
- fusion-level semantic interaction
- target-space semantic prediction
- objective-level semantic alignment
- inference-time semantic retrieval

### 4. Prioritize Slice Evaluation

Overall metrics are necessary but insufficient. Every new method should be evaluated on meaningful slices whenever data allows:

- cold items
- tail items
- short-history users
- sparse users
- metadata-rich versus metadata-poor items
- noisy or truncated semantic content

### 5. Preserve Strong ID-Based Baselines

Semantic methods must always be compared against competitive `ID-only` baselines.

Do not let semantic modeling hide weak sequential modeling.

## Expected Repository Structure

The codebase should gradually evolve toward clear separation of concerns:

- `datasets/`: dataset adapters and processed metadata
- `models/`: benchmarked recommender implementations
- `modules/`: reusable semantic encoders, fusion layers, heads, and losses
- `configs/`: experiment configurations
- `evaluators/`: metric and slice evaluation logic
- `scripts/`: training, preprocessing, and reproduction scripts
- `docs/`: benchmark notes, taxonomy, and experiment records

Adjust names if needed, but keep the separation clean.

## Experiment Policy

- Every benchmark setting must declare both history representation and target space.
- Every semantic method should document what content fields it consumes.
- Every experiment should be reproducible from configuration.
- Every result table should distinguish overall performance from slice performance.
- Every added baseline should be categorized by mechanism, not just by model family.

## Coding Policy

- Keep implementations simple, explicit, and configuration-driven.
- Prefer reusable modules over one-off experiment code.
- Avoid coupling dataset-specific assumptions into generic model code.
- Make semantic features pluggable so the same backbone can run with different content inputs.
- Document hidden assumptions near the code that depends on them.

## Git Workflow

This repository must be maintained with `git`.

For every code change:

1. Make the change.
2. Review the affected files.
3. Run any relevant checks or tests that are available.
4. `git add` the intended files.
5. `git commit` with a clear non-interactive commit message.

Do not leave code modifications uncommitted after completing a task.

Use small, descriptive commits so the benchmark evolution stays easy to track.

## Collaboration Notes for Agents

- Start by understanding whether a task changes benchmark scope, code structure, or experiment logic.
- Prefer extending the benchmark abstraction over adding special-case logic.
- If a new method does not fit the current abstraction, improve the abstraction first.
- Keep paper-style clarity in mind: the repository should help explain the benchmark, not just run it.

## Near-Term Priorities

The first phase of `Secommenders` should likely focus on:

- defining the benchmark taxonomy
- implementing the modular pipeline
- supporting multiple history representations and target spaces independently
- building strong `ID-only` and `semantic-augmented` baselines
- adding slice-aware evaluation
- writing a crisp README and experiment protocol
