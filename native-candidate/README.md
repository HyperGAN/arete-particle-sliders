# Arete native PG0.9.5 source

This separate `arete_native95` project implements the full native Arete27B
training/RAW/finished-text route. **Nine distinct tiny native CPU software cases
are Root-adopted: eight retained passes plus one corrected generation case.**
Only the failed generation case was run again; the eight passing cases were
not replayed. [Portable actual evidence](evidence/cpu-nine-distinct-qualification.json)
retains both attempts, original artifact names/bytes/SHA256, exact versions and
scope. Pretrained27B execution, actual two-GPU fit, trained controls, useful
sliders, effect/LR scores and released weights remain unqualified/absent.

The original root project, PG0.9.4 code, locks, fixtures and evidence stay
byte-identical. The measured native code/tests/locks are source5dfd5459; this
publication updates documentation and portable evidence only.

## Native route and exact sources

Model: [`NousResearch/Arete-27B`](https://huggingface.co/NousResearch/Arete-27B/tree/108f0afbdcfaa1ceeda4a635004304dfff6d4da2),
revision `108f0afbdcfaa1ceeda4a635004304dfff6d4da2`. Its pinned model lock covers
32 files, including18 BF16 shards, total54,743,825,646 bytes. Fresh anonymous
official API metadata confirms public/ungated access and every existing lock
size/LFS SHA/Git blob identity. No pretrained model weights were consumed by these CPU checks; staging is a separate prerequisite.

Serialized architecture is the official `Qwen3_5ForConditionalGeneration`:
64 text layers (48 gated-linear attention +16 full attention), full hidden
width5120, original vocabulary and vision tower. The primary model/config and
[Transformers5.15 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
are used; no older/different model fallback. Only the same304 language attention
Linears are adapted. Vision, MLPs, embeddings, recurrent/convolution parameters,
normalizations and output head stay frozen.

Shared core0.14.1@`41f6066f2d2644902670d5eb736de052e1a70553` owns PG0.9.5,
optimizers, exact FULL/RNG/optimizer restoration and native RAW artifacts. The
product explicitly declares `penalty_input_domain='continuous'`; the training
response is the complete final native AR-token5120 state, with the same frozen
neutral state plus four native context coordinates as conditioning. No width
projection, truncation, field reshape, readout substitute or custom loss occurs.

The new shared routed implementation is explicit in its own source/profile;
there is no dependency on removed historical Fibo/HNDL helpers. It preserves
rank8/alpha8, one original128×4 bank, Kaiming down/zero-up and Arete's original
unit-Normal bank law. This is a new cohort, not a compatibility claim for old
Arete HNDL checkpoints. See [source notices](THIRD-PARTY-NOTICES.md).

## Explicit placement and initialization

`single_placement('cpu'|'cuda:0'|'cuda:1')` gives a declared single-device route.
`two_device_placement()` gives a complete literal map: vision/embed/rotary and
layers0–31 on logicalcuda0; layers32–63, norm and head on logicalcuda1. It uses
whole official decoder/vision boundaries. Root owns actual two-GPU leases,
per-device caps/admission and cleanup; source metadata is not that admission.
There is no auto/balanced placement, quantization, disk or CPU weight offload.

All added FP32 branches, the single bank and critic stay on cuda1. The unchanged
shared `TinyParticleProjection` moves the entire host input to its branch device
and the entire delta back, retaining differentiable copies and original
Parameters. The final native state is copied to the PG device without changing
coordinates. Explicit nested device descriptors ensure core RNG guards cover
both GPUs. The loader checks the actual whole-parameter placement. Native
Accelerate `AlignDevicesHook` wrappers are preserved, not bypassed; only the
exact non-offloading wrapper around a plain original Linear is admitted.

`open_training(config, resume=...)` passes fresh/restore intent before model
construction and optimizers. Fresh construction binds host RMS/zero-up draws to
the declared seed, then applies the explicit per-component QR/R2 helper.
Pretrained parameters are never initialized. Restore validates the original
receipt and does not call that helper. RAW inference restores receipt metadata
and loads the exact current adapter/bank weights, with no teacher/critic/game or
fresh helper. Every weight-EMA path is refused; `ema_decay=None` is required.
Literal `lr`, `d_lr_mult`, `prior_lr_mult` go unchanged to the shared recipe;
defaults are official recipe defaults, never an optimized Arete claim.

## CPU-qualified runtime and native use

`runtime-candidate.json` records the actual CPU-qualified software scope:
Python3.12/Torch2.9.1/Transformers5.15.0/Accelerate1.15.0, exact core41f/PG0.9.5.
The [portable evidence](evidence/cpu-nine-distinct-qualification.json) records
all actual distribution versions. The original prospective runtime plan is
retained as a clearly historical [snapshot](evidence/runtime-plan-before-qualification.json).
Use a separate compatible environment; do not upgrade or relabel historical
prefixes. No old CPU cases/proofs need replaying merely for publication.
The source validator requires the exact core41f API bodies, not merely the
version0.14.1 label. From a fresh Python3.12 environment and this repository:

```sh
python -m pip install 'torch==2.9.1' 'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@41f6066f2d2644902670d5eb736de052e1a70553#subdirectory=packages/particle-sliders-core' './native-candidate'
```

This is a public installation recipe, not an instruction to rerun already
completed qualification or to allocate the full pretrained model. The actual
CPU environment versions are retained in the evidence; fresh platform installs
still require their own dependency review.


The nine cases exercise real small random Qwen3.5 forward/backward/generation,
original Parameter ownership, fresh/restore/RAW mechanics, explicit dispatch
hooks and native5120 continuous admission. They do not execute the full27B or
actual two-GPU model. Exact32-file staging and a separately admitted full native
forward/backward/RAW/text pilot are still required.

```python
from arete_native95.contracts import two_device_placement
from arete_native95.lifecycle import open_training
from particle_sliders.lifecycle import SessionConfig
# Load examples/native-config.json and set actual model/training paths.
# For an owned two-GPU entry, replace options['placement'] with this exact map:
placement = two_device_placement()
```

The installed `arete-native-slider` has `check`, `train` and `generate` commands.
`check` is allocation-free. `examples/native-config.json` and `examples/rows.jsonl`
show literal configuration and matched messages; paths are explicit placeholders,
not private machine paths or invented staged files. `train --config ... --output
NEW_DIRECTORY --stop-after 2` checkpoints every committed update and exports
RAW. It stops at the requested count within the fixed total horizon; no automatic
continuation/promotion exists. Exact continuation supplies the original config
and FULL via `--resume`; RAW cannot resume training.

`generate --artifact RAW.pt --local-options LOCAL.json --request REQUEST.json
--output NEW.txt` uses the original checkpoint tokenizer/template, explicit
thinking choice, native cached autoregressive generation and both native stop
IDs. Local options contain only `model_dir` and the exact `placement`. Output is
actual decoded generated token IDs, with native tags/EOS retained and a metadata
sidecar. A fresh per-call `GenerationConfig` explicitly sets greedy `do_sample=False`,
`use_cache=True`, single beam/sequence and output/length/sampling fields while
retaining the checkpoint native BOS/EOS/PAD IDs. It passes no obsolete
`use_model_defaults` argument. The corrected case checks the resolved config
against hostile inherited defaults; each inference call preserves CPU/all-owned-CUDA RNG.
A context too long is refused, never truncated. OFF/ON differ only by slider
strength using matched seed/request; no response-vector rendering occurs.

The remaining concrete prerequisites are exact32-file model staging and
actual owned two-device fit/forward/backward/RAW/native-text execution. There is no trained/released slider, effect score,
catastrophic qualification or LR winner claim in this source change.
