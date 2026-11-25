# Activation-dependent Arete generation

Use `arete_native_scheduling.generate(session, steering=policy, ...)` on an
already opened native RAW text-render session. Cached policies read the decoder
input before attention and advance once per new input token after prefill.
Answer-only post-block S-PSR reads the selected decoder block's output and
starts at the final prompt position, which predicts the first answer token.
The sidecar records actual cache positions and effective strengths through
native EOS, along with the coefficients or gate decisions.

## Shared equations and archived RAW

The strict native owner verifies its original shared engine and its captured
product Sources. Keep that installed owner and build a separate wheel from the
[pinned shared steering commit](https://github.com/HyperGAN/particle-sliders/pull/269).

```sh
python -m pip wheel --no-deps --wheel-dir steering-wheel \
  'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@b1f780decc516808cfd3019d5e5b674457161f86#subdirectory=packages/particle-sliders-core'
```

Pass this wheel file to `load_steering_core`. It verifies the bytes of both
self-contained steering modules and loads them in a private namespace. The
installed `particle-sliders-core` package still owns the model/adapter
construction and packet validation. Preserve the source-matched RAW factory,
its installed core, the original checkpoint/config and the approved placement.

## Supply a calibrated policy

Choose the intervention and supply a compatible direction or fitted probe, an
evidence reference, the native layer and the calibrated strength.

```python
from arete_native_scheduling import CachedGate, generate, load_steering_core

core = load_steering_core(shared_steering_wheel)
policy = CachedGate(
    core=core,
    layer=calibration["decoder_input_layer"],
    method="cfp",
    parameter=calibration["threshold"],
    intervention="residual",
    direction=calibration["native_direction"],
    preserve_norm=calibration["preserve_norm"],
    projection_bounds=calibration["projection_bounds"],
    evidence_ref=calibration["evidence_ref"],
)
output = generate(session, ids=prompt_ids, max_new_tokens=token_budget,
                  strength=calibration["strength"], steering=policy, seed=seed)
output.save(output_path)
```

- For a direct residual edit, CFP and CLABS steer along the supplied direction
  at the chosen layer with the shared equations. Use
  `core.adaptive.contrastive_sae_direction` with an SAE validated on the same
  residual. CLABS uses a calibrated bias, supplied `CLABSParameters` and an
  explicit controller seed. Use `method="random"` and a matched firing rate for
  the allocation control. Keep the saved adapter inactive in these arms.
  [Gate equations and active budget calibration](https://arxiv.org/html/2611.18274v1)
- For a gated saved adapter, select `intervention="raw_adapter"` and `layer=0`.
  The layer-0 input comes from the current token embeddings and is available
  before all text-attention adapter targets. The gate selects the original
  routed RAW edit at the supplied unipolar strength in `[0,1]`. This tests the
  paper's allocation equations with the product's edit. It requires its own
  layer-0 probe and qualification.
- For answer-only S-PSR, construct `core.prompt.PromptSteering`, load its
  fitted state dictionary and pass
  `PostBlockPromptSteering(core=core, layer=layer, fitted=module,
  evidence_ref=fit_evidence)`. Fit the probe and direction jointly from
  matching decoder block outputs under neutral and attribute prompts with
  the shared layer loss and location penalty. Calibrate the signed global
  strength on separate cases. The `answer_steps` sidecar records output
  index zero at the last prompt position. Use `CachedPromptSteering` to
  replay fits made on decoder inputs after prefill.
  [PSR equations and training](https://arxiv.org/html/2605.03907v1)

### Authored all-layer Arete A-PSR

Load the independently verified authored RAW probe, its exact training plan and
the original RAW8 packet as bytes. The loader binds the artifact digests and
restores all 64 fitted FP32 probes on CPU. Open the base through its original
native backend and placement. For the archived PG 0.11.0 cohort, use the
existing guarded cached runtime and strict RAW8 restore. Bind that owner using
its scope check and native text device context.

```python
from arete_native_scheduling import (
    bind_archived_all_layer_session, generate_all_layer, load_all_layer_prompt_steering,
)

steering, base_packet = load_all_layer_prompt_steering(
    core, probe_bytes=verified_probe_bytes, probe_sha256=verified_probe_sha256,
    training_plan_bytes=verified_plan_bytes, base_raw_bytes=verified_raw8_bytes,
    evidence_ref=evaluation_evidence,
)
session = bind_archived_all_layer_session(
    restored_archived_owner, base_packet=base_packet, check=archive_scope_check,
    render_context=archive_text_device_context,
)
output = generate_all_layer(
    session, steering=steering, strength=calibrated_strength,
    max_tokens=registered_context_limit, max_new_tokens=registered_token_budget, ids=prompt_ids,
    enable_thinking=False, seed=seed,
)
```

The saved adapter runs at zero strength while the shared A-PSR equation edits
each block's output from the final prompt position onward. In native order,
`adaptive_steering.answer_steps` records each output index, cache position,
observed coefficient and effective strength per block. The first forward uses
an empty dynamic cache, the same cache grows by one position per subsequent
forward, and a generated EOS counts as its final output token. Coefficients and
strengths use eight-digit IEEE-754 binary32 hex strings. Zero strength leaves
each block output untouched and records zero effective strengths with null
coefficients. The caller supplies the calibrated strength and the separately
registered evaluation request.

Keep the existing `step_gains` parameter for fixed-schedule replay and select
one mode per request. Use disjoint calibration and evaluation cases. For CFP
and CLABS, use `core.adaptive.calibrate_gate_budget` on steering-active
rollouts. Sum `usage.fired` and `usage.eligible` across the calibration cases
and return pooled `GateUsage`. Report the realized firing fractions separately
on held-out cases. Compare the same prompt, seed and saved RAW against constant
strength and a budget-matched random control. Rank finished text by the
equal-case difference in intended-effect intensity between ON and OFF,
and apply the catastrophic-output check separately. For direct residual
arms, use adapter-OFF controls at the same native layer/request.

The existing cheerful/appreciative garden and coast records identify their
exact source cohorts and frozen effect observations. They contain finished
text and last-token training responses. Native adaptive gains still require a
compatible Arete SAE or aligned per-position prompt-pair activations, plus
disjoint active calibration and finished-output evaluation. The CPU native
fixtures verify the hook, gate/fit execution, cache accounting, OFF and fixed
replay and cleanup.
[Existing evidence and retention rules](finished-output-learning-rates.md)
