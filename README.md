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

MIND also provides nested small-scale variants matching the RVS/RAS scale
protocol. `MINDS<N>` uses a stable user shuffle, a shared 0.3% held-out test
tail, and the first `N%` of all deduplicated users for training. Available
scales are `1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99`.

```bash
python formatter.py --data minds10
python processor.py --data minds10
```

### Shared embedding and quantization runs

Embedding and quantization artifacts use a single-producer lock. If another
trainer, compiler, or quantizer requests the same artifact while it is being
built, it waits instead of launching duplicate work. Waiting processes print
the producer's current stage and progress every five seconds. Machine-readable
progress is stored beside the artifact in `run_state.json`; the lock is removed
after success or failure, and stale local locks can be recovered automatically.

### Dataset statistics

Compare formatted datasets in one command. Missing formatted artifacts are
prepared automatically; use `--no-prepare` to require existing artifacts.

```bash
python scripts/dataset_statistics.py --data ras1,ras2,ras5,ras10,ras20,ras50,ras99
python scripts/dataset_statistics.py --data ras1,ras5,ras99 --output reports/ras_statistics.csv
```

The terminal report includes catalog and observed item counts, user/source-user
counts, interactions, test users, history-length quantiles, density, and
sparsity. It also shows the cold threshold, percentage of observed cold items,
and item retention relative to the observed items at the largest requested
scale. CSV, Markdown, and JSON outputs contain the full set of statistics.

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

To report test performance by the target item's finetune supervision frequency, reuse the same
checkpoint and add:

```bash
python trainer.py ... \
  --test_only true \
  --load_ckpt artifacts/trained/<dataset>/<trained_signature>/<seed>/train/best.pt \
  --frequency_breakdown true \
  --frequency_buckets 0,5,20,100
```

The fixed buckets above are `0`, `1-5`, `6-20`, `21-100`, and `101+`. Frequency counts only
positions used as next-item targets in the existing compiled finetune trajectories; the first
context-only item of each trajectory is excluded. The command reads compiled artifacts without
rebuilding or modifying them and writes the breakdown to
`<test-only-run>/analysis/frequency_breakdown_test.json`.

To run this analysis sequentially for every trained experiment in a scheduler plan:

```bash
python scripts/scheduler_frequency_breakdown.py \
  --plan config/mind_scaling_scheduler.yaml
```

The script reuses each experiment's persisted successful batch size when available, skips plans
whose `best.pt` checkpoint does not exist, streams trainer progress, and prints every completed
experiment's bucket table to the terminal. It also stores per-target raw item
IDs, frequencies, and ranks, plus a plan-level manifest used by downstream
slice analysis. Existing analyses with per-target records are reused; older
bucket-only outputs are regenerated automatically.

After generating per-item transfer quality, join it with all comparable UID/SID
experiments in the scheduler plan:

```bash
python scripts/transfer_frequency_analysis.py \
  --plan config/recif_ads_small_scaling_scheduler.yaml \
  --transfer-quality 'reports/transfer_quality/{data}.parquet' \
  --tq-column 'content_ndcg@20' \
  --metric-k 10 \
  --output reports/transfer_frequency.csv
```

The script pairs UID and SID experiments with the same dataset, backbone, and
history-side semantic additions. It reports macro-averaged UID, SID, and
SID-minus-UID ranking metrics for every frequency bucket crossed with low/high
per-item transfer quality.

### SID Transfer Quality

Measure whether content or SID-prefix neighbors recover behavioral co-occurrence
neighbors. A full-scale dataset can provide the behavioral reference while
representations and candidates remain restricted to the requested subset:

```bash
python cooccurrence_analyzer.py \
  --data ras10 \
  --reference-data ras99 \
  --embedding-model llama3 \
  --sid-coder rqvae \
  --sid-export coll \
  --topk 20 \
  --max-anchors 0 \
  --json-out reports/ras10_transfer_quality.json \
  --per-item-out reports/ras10_transfer_quality.parquet
```

The summary reports content- and SID-neighbor NDCG/Recall against behavioral
PPMI relevance. SID proximity is the longest common code prefix, matching
hierarchical decoding. The per-item output includes local/reference frequency,
content and SID retrieval metrics, and their NDCG difference as quantization
loss. Omit `--reference-data` to measure alignment against the subset's own
behavioral evidence.

For a scale comparison, pass all datasets together and use a shared raw-item
universe. One reference dataset is reused for every scale:

```bash
python cooccurrence_analyzer.py \
  --data ras1,ras5,ras10,ras20,ras40,ras80,ras99 \
  --reference-data ras99 \
  --common-items \
  --max-pairs 0 \
  --embedding-model pretrain-multimodal \
  --sid-coder rqvae \
  --sid-export recon \
  --topk 20 \
  --max-anchors 0 \
  --json-out 'reports/transfer_quality/{data}.json' \
  --per-item-out 'reports/transfer_quality/{data}.parquet'
```

`--reference-data` also accepts an equally sized list when datasets need
different references. If an output path omits `{data}`, the analyzer
automatically appends the dataset name before its extension to prevent
overwriting another result.

### Trained Artifact Registry

`config/trainer.yaml` is the canonical experiment template. It keeps the user-facing schedule arguments small while expanding them into explicit `representation`, `upstreams`, `decoder`, `model`, and `trainer` sections. Signatures are computed from this canonical template, not from user-provided artifact signs.

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

For trained artifacts, the registry initializer also backfills missing canonical upstream defaults from the current template before recomputing signatures. This lets older commands and meta files collide with the new signature when they describe the same experiment under today's explicit defaults.

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

Scratch backbones:

- `scratch` uses a randomly initialized Llama causal Transformer. Its hidden size, layer count, head count, dropout, and maximum length come from the existing scratch configuration, and sequential SID decoding supports KV-cache.
- `scratchlegacy` preserves the former `torch.nn.TransformerEncoder` implementation for reproducing old experiments. Old scratch checkpoints must be loaded with `--model scratchlegacy`.

Running `python scripts/init_artifact_registry.py --stage trained` reports old scratch artifacts as `migration=scratch->scratchlegacy`. Rerun with `--apply` to rewrite their metadata, move them under the scratchlegacy signature, and retain the old signature as an alias. Artifacts carrying the new `backbone_architecture=llama-v1` identity marker remain under `scratch`.

Early stopping accepts one or more main metrics. Separate multiple metrics with `|`, for example `--main_metric 'loss|ndcg@10'`. Loss metrics are minimized, other metrics are maximized, and an improvement in any listed metric resets patience and saves the current checkpoint. Metadata records both the first metric's backward-compatible `best_valid_metric` and the complete `best_valid_metrics` mapping.

The current scheduler uses these rules:

- `batch_size * accumulate_batch = 64`
- `code_beam_chunk_size = max(batch_size, 4 * code_beam_width)` when left at `0`; this bounds active KV-cache beams and normally batches four samples together
- scheduler state persists terminal failures; pass `--retry-failed` when restarting a plan to reset only failed experiments to pending while leaving completed experiments untouched
- initial batch size cap by model name:
  - `scratch -> 64`
  - `scratchlegacy -> 64`
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
