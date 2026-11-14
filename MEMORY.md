# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Default uncheckpointed response training: single GPU | 1 | 420 | 68 | batch 2; ≤ 4096 tokens; BF16 base/FP32 adapters; eager; checkpointing=False |
| Explicit32/32 split: short checkpointed training | 2 | 36/38 | 68 | 2 GPUs; batch 2; ≤ 256 tokens; all 64 blocks checkpointed; all trainables onGPU1 |
| Default finished text: explicit 32/32 split | 2 | 38/40 | 68 | batch 1; ≤ 4096 prompt+128 new; eager; 16full-attention KV layers/48recurrent; no offload |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.9.5 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **NousResearch/Arete-27B**, revision **`108f0afbdcfaa1ceeda4a635004304dfff6d4da2`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Default uncheckpointed response training: single GPU | checkpoint_byte_upper: 50.9560; trainable_transient: 1.5434; saved_activations: 358.0000; recurrent: 0.2959; workspace: 8.0000 |
| Explicit32/32 split: short checkpointed training | per_GPU_frozen_upper: 27.8462; split_activation_upper: 0.7646; per_GPU_recurrent: 0.1479; GPU1_trainable_transient: 1.5434; workspace_per_GPU: 6.0000 |
| Default finished text: explicit 32/32 split | per_GPU_frozen_upper: 27.8462; prefill_per_GPU_upper: 3.8438; KV_per_GPU: 0.1289; GPU1_logits_upper: 1.8945; workspace_per_GPU: 6.0000 |

Detailed arithmetic and assumptions:

- **whole BF16 checkpoint bytes including headers**: `54713605792`.
- **LoRA elements**: `23760896`.
- **projections**: `304`.
- **generator plus critic elements**: `25894049`.
- **split frozen formula**: `F/2 +2*248320*5120 bytes on EACH GPU: deliberately reserves full embedding/head-sized placement imbalance rather than claiming perfect balance`.
- **host loading**: `complete BF16 checkpoint +two largest3,988,973,120-byte shard buffers +8 GiB +trainable export`.

Scope and limits:

- The large default 4096-token/uncheckpointed estimate is intentional: fitting frozen weights alone does not bound backpropagation memory. The short checkpointed two-GPU profile is a separate explicit option, not an assertion that defaults already enable checkpointing.
- Only16of64 layers use full KV attention.48linear layers have FP32 recurrent states; the ledger includes their state/convolution buffers and does not pretend all 64 layers have a transformer KV cache.
- All304native hybrid attention Linears are adapted; vision/MLP/convolution/head remain frozen. No quantization or CPU-offload fallback is accepted by the native profile. Full checkpoint file bytes are an auditable conservative weight upper bound; no invented exact parameter count is used.

Source facts (read as text; no model execution):

- [model-source/config.json](model-source/config.json)
- [native-candidate/src/arete_native95/model.lock.json](native-candidate/src/arete_native95/model.lock.json)
- [native-candidate/src/arete_native95/contracts.py](native-candidate/src/arete_native95/contracts.py)
- [native-candidate/src/arete_native95/native.py](native-candidate/src/arete_native95/native.py)
- [native-candidate/src/arete_native95/targets.py](native-candidate/src/arete_native95/targets.py)
- [native-candidate/src/arete_native95/host_source.py](native-candidate/src/arete_native95/host_source.py)
- [Pinned model metadata](https://huggingface.co/NousResearch/Arete-27B/tree/108f0afbdcfaa1ceeda4a635004304dfff6d4da2) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.9.5 release `1b8355b184b21bb7d81104ee590813c262f6df0a`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `pg09_audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
