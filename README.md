# Secommenders

`Secommenders` is a benchmark for semantic-aware sequential recommendation.

The repository is organized around a configurable pipeline:

- `formatter.py` / `formatters/`: dataset formatting
- `processor.py` / `processors/`: train/valid/test split preparation
- `embedder.py` / `embedders/`: item content embedding extraction
- `quantizer.py`: semantic code generation such as SID/hash assets
- `clusterer.py`: hierarchical UID decoding assets from user sequences
- `compiler.py`: compiled training/evaluation artifacts
- `trainer.py`: model training, validation, and testing
- `scheduler.py`: multi-GPU batch experiment scheduler

## Installation

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install that first and then run the command above again.

## Common Entry Points

### Format / Process

```bash
python formatter.py --data mind
python processor.py --data mind
```

### Content Embeddings

```bash
python embedder.py --data mind --model qwen3embedding4b
```

### Semantic Codes

```bash
python quantizer.py --data mind --model llama3 --quantizer_name rqvae
python quantizer.py --data mind --model llama3 --quantizer_name lsh
```

### Hierarchical UID Assets

```bash
python clusterer.py --data mind --uid_cluster_levels 20,20
python clusterer.py --data mind --uid_cluster_levels auto,auto
python clusterer.py --data mind --uid_cluster_levels auto/10,auto/10
```

`uid_cluster_levels` examples:

- `10`: one cluster layer with 10 clusters
- `20,20`: two cluster layers, each with 20 clusters
- `auto,auto`: two cluster layers, each resolved automatically from item count
- `auto/10,auto/10`: same as `auto`, then divided by 10

## Trainer Notes

### Valid-Only

`valid_only` supports both smoke-check and full validation:

```bash
python trainer.py ... --valid_only 1
python trainer.py ... --valid_only true
```

- `--valid_only 1`: run only the first validation batch
- `--valid_only true`: run the full validation set

### Test-Only with Checkpoint

```bash
python trainer.py ... \
  --test_only true \
  --load_ckpt artifacts/trained/<dataset>/<trained_signature>/<seed>/train/best.pt
```

This is useful when training succeeds but final test evaluation needs a smaller batch size or safer decoding settings.

### Trained Artifact Registry

Trained artifacts are stored by evaluation setting, seed, and execution phase:

```text
artifacts/trained/<dataset>/<trained_signature>/<seed>/<phase>/
```

`phase` is one of `precheck`, `train`, or `test`. `valid_only` smoke runs write to `precheck/`, full training writes checkpoints and final metrics to `train/`, and test-only fallback runs write to `test/`.

The trained signature intentionally excludes `seed`, `batch_size`, and `accumulate_batch`. It includes their product as `effective_batch_size`, so `batch_size=64, accumulate_batch=1` and `batch_size=32, accumulate_batch=2` resolve to the same trained setting.

Trained runs are also resolved through a dataset-level registry before a new run directory is created:

```text
artifacts/trained/<dataset>/.index.json
```

This lets newer config signatures reuse older folders when a signature schema changes but the old run can be migrated by filling in default hyperparameters.

Initialize or inspect the registry for existing runs with:

```bash
python scripts/init_artifact_registry.py --stage clustered
python scripts/init_artifact_registry.py --stage compiled
python scripts/init_artifact_registry.py --stage quantized
python scripts/init_artifact_registry.py --stage trained
python scripts/init_artifact_registry.py --stage trained --apply
```

The dry run reports unresolved folders and delete candidates. A folder is only migrated when its `meta.json` contains enough information to rebuild the current artifact spec.

`clustered`, `compiled`, and `quantized` use the same dataset-level `.index.json` alias registry as `trained`, but they do not create seed or phase subdirectories. Their registry maps the latest signature back to the existing artifact folder. For `quantized`, the artifact folder is the quantizer root, e.g. `artifacts/quantized/<dataset>/<embedding_model>/<quantizer_variant>/`; checkpoint folders such as `best`, `best-recon`, `best-usage`, `final`, and `exports` remain internal subdirectories.

The same script is also the migration path from older trained layouts. It upgrades both legacy flat folders and the previous `<trained_signature>/<seed>/` layout into the current `<trained_signature>/<seed>/<phase>/` layout, so it is safe to rerun after an earlier registry initialization.

If migration reports a conflict, the old folder and canonical target folder both contain results for the same trained signature and seed. Inspect both sides first:

```bash
python scripts/init_artifact_registry.py --stage trained --apply --json
```

Then resolve them interactively:

```bash
# Prompt for each conflict, apply the choice immediately, then continue to the next conflict.
python scripts/init_artifact_registry.py --stage trained --apply --interactive
```

For each conflict, choose target to keep the canonical target and delete the source, or choose source to delete the target and move the source into place. If only one side appears to contain results, the prompt suggests that side.

If you have reviewed the dry-run output and want to remove all precheck/valid-only runs plus abnormal train runs that have no `best.pt`, pass the explicit deletion flag:

```bash
python scripts/init_artifact_registry.py --stage trained --apply --delete-abnormal-empty
```

### Hierarchical UID Decoding

Enable hierarchical UID decoding with:

```bash
python trainer.py ... \
  --task_type uid \
  --uid_decoding hierarchical \
  --uid_cluster_levels 20,20 \
  --uid_cluster_topk 5,5,20
```

`uid_cluster_topk` must provide one value per hierarchy depth:

- one cluster layer (`10`) -> depth `2` -> for example `3,20`
- two cluster layers (`20,20`) -> depth `3` -> for example `5,5,20`

## Batch Scheduler

The scheduler is designed for running many experiments across multiple GPUs while handling OOM automatically.

### Plan File

Example plan:

- [config/scheduler.example.yaml](/Users/jyonn/Projects/Research/Secommenders/config/scheduler.example.yaml)

Run it with:

```bash
python scheduler.py --plan config/scheduler.example.yaml
```

Each experiment can be written in either of two forms:

1. Structured args

```yaml
- name: mind_qwen_sid2sid
  args:
    data: mind
    model: qwen35th4b
    task_type: sid
    repr_type: sid
    repr_source_model: llama3
    sid_coder: rqvae
    sid_export: recon
```

2. Raw trainer command

```yaml
- name: mind_qwen_sid2sid
  command: >
    python trainer.py --data mind --model qwen35th4b --task_type sid
    --repr_type sid --repr_source_model llama3 --sid_coder rqvae
    --sid_export recon
```

### Scheduler Rules

The current scheduler uses these rules:

- `batch_size * accumulate_batch = 64`
- `code_beam_chunk_size = batch_size`
- initial batch size cap by model name:
  - `scratch -> 64`
  - `qwen35th08b -> 32`
  - `qwen35th4b -> 16`
  - `llama3 -> 4`
  - `qwen35th9b -> 4`
- GPU launch thresholds by free memory:
  - `scratch -> 10G`
  - `qwen35th08b -> 20G`
  - `qwen35th4b -> 40G`
  - `llama3/qwen35th9b -> 80G`
- if the experiment uses `embedding` in repr or task:
  - thresholds are shifted up by one tier
  - `scratch -> 20G`
  - `qwen35th08b -> 40G`
  - `qwen35th4b -> 80G`
  - `llama3/qwen35th9b -> 80G`
- only `sid/hash` experiments run OOM precheck
  - precheck command is `--valid_only 1`
- `uid` and other tasks start training directly
- if training or precheck OOMs, the scheduler backs off along:
  - `64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1`
- if training succeeds but final test OOMs:
  - the scheduler retries with `--test_only true --load_ckpt ...`
- if the resolved trained run directory's `meta.json` already contains `test_metrics`,
  - the experiment is skipped as an existing completed run
- multiple seeds of the same setting share the same remote evaluation signature
  - each seed remains a separate experiment under that evaluation

### Scheduler Outputs

The scheduler writes its own state under:

```text
artifacts/scheduler/<plan_name>/
```

This includes:

- `state.json`: experiment states and retry history
- `logs/`: per-launch stdout/stderr logs

## Current Status

The benchmark is still evolving. The most actively developed paths right now are:

- semantic target spaces such as SID/hash
- semantic-augmented history representations
- hierarchical UID decoding
- batch experiment automation
