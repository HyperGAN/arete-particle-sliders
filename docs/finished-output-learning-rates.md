# Arete: saved RAW32 best-tested rate

As of 2026-11-09, **G0.012 is best tested** for cheerful, appreciative tone:
mean intended-effect gain **+1.0** versus **0.0** for G0.001. Both arms completed
32 updates at horizon128/batch2/rank8, seed7, the same128×4 bank, full native5120
response and matched garden/coast text generation (greedy, up to64 new tokens,
native EOS, strength0 OFF /1 ON). Actual D stays0.018000000000000002 and bank0.03.
This is a completed **provisional model-instrument** comparison;
`trait_qualified:false` remains explicit. Use the saved G0.012 RAW32 within this
scope; this is not a global optimum, semantic validation or changed default.

| G | Garden OFF → ON | Coast OFF → ON | Equal-case mean gain |
|---|---|---|---|
| **0.012** | 3 → 4 | 0 → 1 | **1.0** |
| 0.001 | 3 → 3 | 0 → 0 | 0.0 |

These are the frozen instrument's original **raw0–4 intensity units**, not
probabilities or normalized0–1 scores. The two saved OFF firsts were reused
without rerating. Each genuine FULL32 candidate separately passes its qualified
catastrophic guard. That guard measures catastrophic failure, not cheerfulness;
its pass does not provide formal trait qualification. The earlier RAW2 one-case
gain0 is a separate checkpoint/budget, not this comparison.

[The portable summary](finished-output-rate.json) preserves all levels, exact
rates, checkpoint metadata and retained decision/result/guard hashes. Full private
RAW/FULL files, teacher cache and first-response records are not public downloads
in this repository. Their identities support scalar arithmetic reproduction;
this documentation does not bundle an executable pretrained run or release weights.

## Actual options and use

The current native project is [native-candidate](../native-candidate/README.md),
namespace `arete_native95`, PG0.9.5/core0.14.1 at the captured core41f source.
The root PG0.9.4 CPU fixture remains historical and separate. Add these literal
members to `options` in a **new, explicitly declared native cohort** based on
[native-config.json](../native-candidate/examples/native-config.json):

```json
{"lr": 0.012, "d_lr_mult": 1.5, "prior_lr_mult": 2.5, "ema_decay": null}
```

`lr` is G; D is `lr*d_lr_mult`; bank is `lr*prior_lr_mult`. These exact literals
are passed through `ConditionalPG095(recipe_overrides=...)`, preserving the
shared official objective, penalties and optimizer formula. For the compared
G0.001 arm the saved literals are0.001/18.0/30.0, giving the same D/bank.
Do not reconstruct rates from rounded quotients. Set `options.steps=128` for
the fixed horizon and use `--stop-after 32` only for the declared32-update budget;
that does not change the horizon to32. Preserve literal batch2/seed7, native
geometry, source profiles, explicit owned placement and authenticated model/cache.
These choices describe this comparison, not instructions to repeat it.

The existing product entry points accept:

```sh
arete-native-slider train --config YOUR-NATIVE-CONFIG.json --output NEW-RUN --stop-after 32
arete-native-slider train --config SAME-NATIVE-CONFIG.json --output NEW-CONTINUATION --resume EXACT-LATEST-FULL.pt --stop-after 32
arete-native-slider generate --artifact SOURCE-MATCHED-RAW.pt --local-options LOCAL.json --request REQUEST.json --output NEW.txt
```

The continuation form applies only to genuinely missing updates, never a completed
RAW32. Use the same complete config and original source-bound FULL (game, both
optimizers, bank and RNG); RAW is inference-only. The CLI requires a fresh output
directory even for continuation. `LOCAL.json` contains only the actual `model_dir`
and exact `placement`; `REQUEST.json` contains the native messages/request,
strength, seed, max_tokens/max_new_tokens and thinking choice. See the actual
[CLI](../native-candidate/src/arete_native95/cli.py) and
[finished-text validator](../native-candidate/src/arete_native95/finished_text.py).
There is no implicit GPU admission, auto device map or ordinary hosted Generate.
Saved research packets retain their captured factory; changing a factory/config
is not a migration or permission to retag old exports.

Fresh `open_training(config)` applies the declared QR/R2 policy to owned added
trainables **before optimizers**, preserving host RMS, the existing unit-Normal
bank, zero-UP, original Parameters and frozen pretrained model. Explicit
`open_training(config, resume=FULL)` authenticates and reuses its receipt without
initializing again; RAW inference never initializes. No weight EMA is allowed
for adapter, critic or bank. Official scalar optimizer statistics remain unchanged.

## Sole ranking statistic and reuse

Subtract matched OFF intended-effect intensity from ON for each finished case,
then average with equal case weights. Higher wins. Preserve negatives, zero and
exact ties; missing judgments remain missing, never synthetic zero. Teacher/student
SSE is diagnostic only, not a selector or tiebreak. Beauty, preservation, confidence
and weak effect are not additional ranking gates. The separate catastrophic-only
gate stops failed candidates and treats unknown judgments as unmeasured.

Reuse completed searches, saved RAW/FULL states, teacher firsts, finished outputs
and frozen effect observations. No fresh confirmation, repeated comparison or
worse-arm completion is required. Additional training, another concept, budget or
sampling profile is a new scope and cannot inherit these scores or guards.
