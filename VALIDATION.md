# Validation — 2026-11-08

Final installed-package suite: **22 passed, 0 failed, 0 skipped** in 19.96s.
Raw pytest results are in `validation.xml`. No pretrained weights were loaded.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ARETE_TOKENIZER_DIR=/tmp/arete-tokenizer \
.venv/bin/python -m pytest -q --junitxml=validation.xml
```

The official tokenizer test needs the tokenizer files from immutable revision
`108f0afbdcfaa1ceeda4a635004304dfff6d4da2`. Without that explicit directory the
test skips, rather than downloading data. This run staged tokenizer.json and its
small config/template files only; their hashes and rendered prompts are in
`tokenizer-validation.json`.

Runtime: Python 3.12.15, Torch 2.9.1, Transformers 5.15.0,
particle-sliders-core 0.12.0 at `5d68473a098801ee1ab3ebbb12463daef51c6346`,
ParticleGAN 0.9.4, HyperGAN 2.0.0 at the core's verified builder commit,
HNDL 0.10.0. CPU math threads were limited to 1; CUDA was hidden. The working local
venv borrows dependency directories read-only; tests imported the installed
Arete package, with no `PYTHONPATH` override on the final run.

## Passed checks

- Actual Transformers conditional Qwen3.5 fixtures at widths 16 and 32, each with
  three gated linear-attention layers, one full-attention layer and a vision tower.
- All 19 expected fixture projections are wrapped, with no vision/MLP targets.
  Repeated attachment fails before mutation; close restores hooks and flags.
- Shared ParticleGAN updates change the one real routed particle bank. Every
  adapter parameter has a finite gradient; the bank has a nonzero gradient.
- All base tensors remain bitwise unchanged and frozen, with no base gradients.
  Strength 0 remains bitwise equal to its original response after training.
- Four-step uninterrupted training is exactly equal to 2+2 checkpoint/resume,
  recursively including parameters, optimizer state, counters and RNG state.
- RAW portable inference reload matches training-session responses exactly at
  strengths 0,−0.5,+0.5,+1. EMA configuration and EMA export are rejected.
- Shared checkpoint wrappers retain strength through backward recomputation
  after the surrounding scale context restores 0; outputs and all parameter
  gradients match the uncheckpointed path. Checkpoint hooks clean up.
- Changed conditioning is rejected on resume; changed product source hashes
  are rejected before binding. Invalid runtime options fail before model loading.
- The actual full 64-layer Arete architecture is constructed with meta tensors;
  every one of 304 expected projection weights exists in the official index.
- Official tokenizer tests verify native thinking-off/on endings, explicit list
  output, pad ID 248044 and EOS ID 248046.
- Native cached autoregressive generation on the random fixture keeps the RAW
  branch active, restores its scale, and preserves the generation config.
- Missing staged pretrained files are rejected before weight loading.
- The installed shared `particle-slider` CLI accepts and validates the registered
  `arete-cpu-fixture` profile with explicit CPU execution.

## Shared CLI execution

A separate completed four-step **CPU fixture** run used the installed shared
CLI, with `checkpointing=true`, checkpoints every 2 steps, RAW selection and v2
inference export. Its request, result and metrics are retained in `evidence/`;
larger disposable fixture checkpoint tensors remain at
`/tmp/arete-shared-cli-fixture-v1/artifacts/`.

```python
from particle_sliders.session_cli import main
main(['train', '--config', 'examples/fixture.json',
      '--profile', 'arete-cpu-fixture', '--device', 'cpu', '--no-reporting',
      '--out', '/tmp/arete-shared-cli-fixture-v1',
      '--selection', 'raw', '--export-format', 'v2',
      '--checkpoint-every', '2'])
```

Completed update count: 4/4. This run is not an LR search, native 27B training or a
quality result. Use a fresh output directory if repeating a fixture run.

## Corrected during development

The first suite had 14 passes and 2 failures because the tests expected generic
`ValueError` where the shared interface correctly raised `UnsupportedOperation`
and `CheckpointError`. Those expectations were corrected. A separate official
tokenizer check then exposed Transformers5's dictionary return default; the
binding now explicitly requests `return_dict=False`, covered by the final suite.
Shared CLI planning also revealed that the runner supplies an explicit `device`
option; CPU validation and a rejection test for CUDA were added.

## Not run and pending

- Full 27B pretrained source staging/weight hashing, loading, forward, backward,
  cached generation, checkpoint/resume and memory measurement.
- Native-width 5120 training with the supported continuous-domain PG source
  profile. The current registered product profile intentionally refuses native
  training; its fixture profile cannot silently stand in for that native cohort.
- GPU admission, process/lease/resource checks or owner coordination for any
  proposed GPU run. No GPU request was submitted or denied, and no GPU workload
  was started, stopped or modified.
- Learned intended-effect measurement, held-out language quality, LR search,
  long-context behavior, multimodal training, quantization and device sharding.
- This historical validation did not publish code, weights or a deployment.
  The later public source snapshot is documented separately in PUBLICATION.md.

The next native step is a separately reviewed and source-pinned qualification
phase with fresh scheduler admission. Preserve unrelated workload priorities,
training stops, containment holds and existing leases. No current capacity was
inferred from this task's CPU work or from old receipts.
