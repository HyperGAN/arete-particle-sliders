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

#### Compare timing over a registered answer prefix

Use the same registered case, prompt, native seed, token budget and eligible
answer prefix for both arms. Calibrate on a separate split with that same
prefix. The final prefill position predicts the first generated token and
counts as the first eligible position. Both arms retain native EOS and apply
zero steering after the registered prefix.

```python
from arete_native_scheduling import (
    generate_all_layer, generate_all_layer_constant, load_dose_matcher,
    match_all_layer_constant,
)

matcher = load_dose_matcher(current_shared_core_wheel)
request = dict(ids=prompt_ids, seed=seed, max_tokens=registered_context_limit,
               max_new_tokens=registered_token_budget, case_ref=registered_case)
adaptive = generate_all_layer(
    session, steering=steering, strength=calibrated_strength,
    eligible_tokens=registered_answer_prefix, **request,
)
trace = adaptive.native["adaptive_steering"]
if trace["eligible_prefix_complete"]:
    constant = match_all_layer_constant(trace, matcher=matcher)
    control = generate_all_layer_constant(session, steering=steering, constant=constant, **request)
```

The matcher verifies the `strength_schedule.py` source from the [shared core
revision](https://github.com/HyperGAN/particle-sliders/commit/c4b3655f43ee9b6e9780bb65766dee0d603cb749).
The earlier steering wheel still loads the original A-PSR equations. Each
constant uses the nearest representable binary32 gain on its own fixed layer
direction. The receipt records the exact signed adaptive and rendered
constant doses, and the constant-minus-adaptive residual for each layer.
`exact_dose_equal` reports equality across all layers after replay. A
`nearest_binary32` result has a nonzero residual. Coefficient dose is measured
before the residual is rounded to BF16.

An arm that ends before the registered prefix retains its finished text and
records incomplete support. It cannot establish the matched-prefix comparison.
EOS after the prefix can occur at different positions in the two arms.

#### Hand finished firsts to the existing Arete observers

Use `arete_native_scheduling.observer_ingress.prepare_blinded_trio` after the
registered native firsts settle. The caller supplies pinned `path`, `bytes`
and `sha256` references for the registration and each available arm first,
plus the same verified dose matcher used to prepare the constant.

The `arete.a_psr.prefix_observer_registration.v1` record binds the expected
native `prefix_dose_scope` as `scope`, the human-readable `prompt`, the
registered `adaptive_strength`, `ema_decay=None`, and the registered
`dose_policy` of `exact` or `nearest_binary32`. Include the original
`integrity_output_contract` and the existing `observer_binding` containing
`calibration_plan`, `tone_qualification` and `guarded_calibration` references.
The handoff pins the adopted v3 plan, qualification result, and guard by size
and SHA-256. The existing observer guard verifies their bodies in its own
authorized read scope, including the plan's separate integrity qualification.
Each of `off`, `adaptive` and `constant` has distinct `claim_path`,
`first_path`, `text_path`, `native_path` and a preregistered opaque `blind_id`.

Run each authorized arm through its existing guarded native RAW owner. Supply
the exact complete native input IDs for the registered case. The registration
uses the original writer's sidecar path `<text_path>.native.json`.

```python
from arete_native_scheduling import render_prefix_first

first_ref = render_prefix_first(
    registration_ref, arm, session, steering=steering,
    ids=registered_native_ids, matcher=matcher,
)
```

The entry point checks the native case and clock before writing its exclusive
durable `arete.a_psr.prefix_native_intent.v1`. The original native text writer
saves the finished text and sidecar. The entry point then records their
immutable `claim`, `text` and `native` references at `first_path`.
The caller serializes use of its native owner and controls access to the
registered input and output paths. Previously settled firsts return their
verified references. An entered arm with an unresolved first is retained and
raises on a later call.

OFF uses explicit zero on the same registered clock. Constant uses the
verified completed adaptive prefix and returns `None` when early adaptive
EOS left that prefix incomplete. The finished adaptive first is available
for the handoff. The handoff recomputes the exact constant receipt from the
observed per-layer gains and retains valid early-EOS and missing-arm cases
with unmeasured timing.

Pass the projected `tone_bank` rows to the original v3 tone scorer. It masks
reported wording only in its internal scoring view. Pass `integrity_inputs`
to the qualified separate whole-text integrity observer. Both reference the
same untouched finished text. Keep `private` out of the scorer input. Its
`ready_for_observation` timing status concerns firsts and dose arithmetic.
The observed outcome still requires the separately guarded instruments.
Measure the intended effect as the equal-case mean of adaptive tone minus
matched OFF tone. Keep the categorical catastrophic verdict separate.

#### Read development evidence and freeze alpha

In the authorized downstream CPU process, use
`arete_native_scheduling.development_evidence` with the current shared core.
`read_prefix_development_case` joins the original guarded
`blinded-firsts.json` and `guard-terminal.json` with the registered native
firsts. It checks the v3 tone call's blind ID and full-text reference, and the
separate integrity call's complete-text content, contract and qualified
profile. Pass the exact guarded artifacts so the reader can verify their
chronology and identity.

The independent `arete.a_psr.prefix_development_registration.v1` declares
`case_role="development"`, `ema_decay=None`, the exact `case_ids`, and
`trials` containing each measured candidate `alpha` and its mapping of case
IDs to pinned `registrations`. Register the full grid before any native arm
enters. The caller establishes case custody before registration.

`reduce_prefix_development(panel_ref, evidence, matcher=matcher)` consumes
`evidence[alpha][case_id]` with `native_firsts`, `blinded_firsts` and
`guard_terminal` references. Missing pairs stay unmeasured. The shared
`matched_effect_mean` computes intensity over the complete declared roster.
The receipt retains measured intensity when the timing or integrity status
prevents selection. Matched constant has a separate status and mean. The
reader uses the registered RAW identity in the receipts and operates on
settled observer evidence. [Shared reducer in core PR #336](https://github.com/HyperGAN/particle-sliders/pull/336) ·
[Exact merged core source](https://github.com/HyperGAN/particle-sliders/blob/637880b9b40a231a16f2e02ad384738052eca6e0/packages/particle-sliders-core/src/particle_sliders/alpha_search.py).

`freeze_prefix_development(panel_ref, evidence, output_path, matcher=matcher)`
writes an exclusive first receipt for a unique eligible alpha with positive
measured adaptive ON−OFF on development. A tie or nonpositive best result
remains in the returned development evidence. The independent held-out
registration binds the frozen selection before entering its native calls.

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
