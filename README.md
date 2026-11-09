# Arete particle sliders

**Experimental source publication.** Random CPU architecture and tokenizer/software fixtures only. Historical installed suite:22 passed. Full pretrained 27B staging/forward/backward/generation/resume, width5120 profile admission, GPU fit and quality remain unqualified.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Isolated local candidate for `HyperGAN/arete-particle-sliders`, integrating
[NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B) with the
existing shared particle-slider response, training and inference interfaces.
No remote repository, release, deployment or model weights were created.

The qualified executable profile is a **small random CPU Qwen3.5 architecture
fixture**, not the pretrained 27B model. Native training refuses before model
allocation. Full-model source staging, numerical qualification and scheduler
admission remain prerequisites. These checks establish implementation mechanics;
they do not establish a useful learned effect or text quality.

The product adapts only the host-specific architecture and targets. The pinned
shared core owns the routed HNDL projection, one shared 128×4 particle bank,
rank8/alpha8 branches, ParticleGAN game, optimizers, checkpoint lifecycle and
portable inference artifacts. No shared-core code or existing model product was
modified. The core dependency is the established immutable cohort
`5d68473a098801ee1ab3ebbb12463daef51c6346`, version 0.12.0 with ParticleGAN 0.9.4,
HyperGAN's verified HNDL builder and Transformers 5.15.0.

## Use the qualified fixture

Install into an isolated Python 3.11–3.14 environment with a supported Torch build:

```bash
python -m venv .venv
.venv/bin/python -m pip install '.[test]'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The exact local validation runtime used Python 3.12.15, Torch 2.9.1 and
Transformers 5.15.0. The working `.venv` in this checkout borrows existing installed
dependencies through a local `.pth`; it does not change the donor environments.
For a portable installation, install from the pinned dependencies above.

```python
from arete_particle_sliders.binding import config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

rows = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]
with open_response(config(fixture=True, fixture_width=16,
                          steps=4, checkpointing=True), rows=rows) as session:
    session.train()
    session.checkpoint('training.pt')
    artifact = session.export_inference(selection='raw')
with open_inference(artifact, operation='response') as inference:
    response = inference.response(ids=[1, 2], strength=0.5)
```

The installed `arete-cpu-fixture` training profile is discoverable through the
shared `particle-slider` command and supports periodic checkpoints and exact
resume. `examples/fixture.json` provides explicit token rows. These are numerical
hidden-state responses, not a claim of generated-language quality. The positive
teacher receives positive context; the student always receives neutral context.
Signed strengths are mechanically supported; strength 0 bypasses every branch
exactly. Strength calibration remains unmeasured.

**RAW only:** `ema_decay=None` is explicit by default. Any attempt to enable
weight EMA is rejected before model allocation. Current raw adapters and particle
weights are used for training, evaluation, exports and inference.

## Architecture and native loading

The pinned model revision is `108f0afbdcfaa1ceeda4a635004304dfff6d4da2`.
Its actual config is `Qwen3_5ForConditionalGeneration`, with a vision tower and
a 64-layer text decoder:48 gated linear-attention and 16 full-attention layers.
The model card calls the starting model Qwen3.8; the serialized architecture is
Qwen3.5. Do not use a plain Qwen3 attention adapter or assume ordinary q/k/v
projections in every layer.

The target walk validates all layers before changing any forwards. It wraps 304
unique language-only Linears:

- Full attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Gated linear attention: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`,
  `out_proj`.

Vision, MLPs, embeddings, output head, normalization, convolution and recurrent
state parameters stay frozen. Float32 branch parameters remain separate from a
potential bf16 base. Attach/close restores original forwards, modes and gradient
flags; repeated attachment fails without modifying an existing slider.

`backend.verify_staged(path)` streams hashes for every official snapshot file,
including all 18 shards, against `model.lock.json`. `backend.load_staged(path)`
performs this verification and uses the official conditional-generation class,
local files only, `trust_remote_code=False`, eager attention and explicit dtype.
No quantized loading, device sharding or pretrained execution is qualified here.
The loader is an explicit utility; the fixture training profile never calls it.

`backend.encode_chat` uses the checkpoint's own tokenizer/template, explicitly
selects thinking, requests token IDs rather than Transformers5's default mapping,
and adds the native generation prompt. Thinking defaults to off. Native pad ID
is 248044 and generation stops on 248046 or 248044. `backend.generate_raw` keeps
the current raw branch active throughout native cached autoregressive generation
and preserves the base generation configuration. It was checked on the random
architecture fixture, not on pretrained 27B weights.

## Evidence and remaining work

See [VALIDATION.md](VALIDATION.md) for exact passed checks, earlier corrected test
failures, commands and exclusions. The full native config was constructed on
meta and all 304 target names matched the official safetensors index. No actual
27B parameter tensor was allocated. The official tokenizer and both thinking
modes were separately exercised; tokenizer files are not bundled in this repo.

The official checkpoint has 27,356,728,560 bf16 parameters:
54,713,457,120 tensor bytes (about50.96GiB), before runtime memory, activations,
attention/recurrent caches, adapter gradients, critic and optimizer state.
An apparent free GPU or an old scheduler receipt is not admission. No capacity
claim or training-memory estimate has been validated by execution.

Native hidden width 5120 also requires the shared core's explicitly supported
continuous-domain ParticleGAN source profile. The fixture release runtime must
not truncate or reshape native hidden states to bypass that gate. Before
enabling a native training cohort: stage and verify the full pinned snapshot,
qualify that exact source/profile runtime, check native forward/backward and RAW
save/reload numerics, obtain fresh admission from the deployment owner and existing scheduler, then run
bounded qualification. Preserve all unrelated workload priorities, stops, holds
and existing leases; this source publication grants no execution admission.

This repository publishes source only. No pretrained training, learned-slider
quality evaluation, LR search, deployment or weight publication is established.
See PUBLICATION.md for the sanitized snapshot lineage and validation boundary.

## Sources and licenses

Product code is MIT, adapted from the established local Gemma binding; the
original notice is retained in `LICENSE` and adaptation details in
`PROVENANCE.json`. The model is Apache-2.0. Its original license, model card,
config, generation config, chat template, tokenizer metadata and weight index are
preserved in `model-source/`; the repository listing contained no separate
NOTICE file. Model weights and datasets are not redistributed.
