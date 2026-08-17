# Trainer configuration profiles

`trainer.v4` separates representation declarations from encoder and decoder selection. The shared
`_catalog.yaml` declares reusable UID, SID, embedding, hash, and text instances; declarations are
inactive until a profile references them.

Profiles compose shared configuration through RefConfig's native `$$import` directive. Import paths
are relative to the YAML file containing the directive, and profile values recursively override the
imported values.

- `uid.yaml`: UID-only baseline.
- `uid-hierarchical.yaml`: hierarchical UID decoding.
- `sid-content.yaml`: SID learned from content embeddings.
- `sid-collaborative.yaml`: SID learned from Word2Vec collaborative embeddings.
- `sid-hybrid.yaml`: SID generated from content and collaborative embeddings.
- `sid-multi.yaml`: jointly trains independent content and collaborative SID spaces.
- `embedding-dual.yaml`: UID target with content and collaborative embeddings as separate inputs.
- `hybrid.yaml`: UID target with SID, content embedding, and collaborative embedding as separate encoder inputs.
- `hybrid-multi-sid.yaml`: UID target with two independent SID and two independent embedding inputs.
- `multi-decoder.yaml`: joint SID and UID decoding.
- `hash.yaml`: content-derived hash target.
- `uid-text.yaml`: UID target with raw text context; requires an LLM backbone.

Only representations referenced by `encoder.representations` or `decoder.targets` participate in artifact SIGNs.
Each active representation receives an independent `<repr:NAME>` marker. Run a profile with:

```bash
python trainer.py --config config/trainer/hybrid.yaml --data mindf --model scratch
```

Profile placeholders remain regular CLI parameters. For example:

```bash
python trainer.py --config config/trainer/sid-hybrid.yaml \
  --data mindf --model scratch \
  --content_embedding_model llama3 \
  --content_embedding_dim 256 \
  --collaborative_embedding_dim 64 \
  --sid_codebook_size 128
```

The current runtime supports any number of active `sid` and `embedding` instances. Every SID owns
its quantizer upstream, compiled vocabulary and item view, model embedding/head, decoding settings,
and named metrics. Multiple SID decoder targets are trained jointly; their unprefixed evaluation
metrics are the mean across SID targets. UID, hash, and text currently allow one active instance of
each type.
