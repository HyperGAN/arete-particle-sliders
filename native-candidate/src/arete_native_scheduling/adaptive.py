"""Activation-dependent control for the original Arete cached RAW renderer."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal

import torch

from .steering_core import CORE_REVISION, SteeringCore, require_steering_core


@dataclass(frozen=True, kw_only=True)
class CachedGate:
    """One caller-calibrated gate on a native decoder-input residual."""

    core: SteeringCore
    layer: int
    method: Literal["cfp", "clabs", "random"]
    parameter: float
    intervention: Literal["residual", "raw_adapter"]
    evidence_ref: str
    direction: torch.Tensor | None = None
    seed: int | None = None
    clabs: object = None
    preserve_norm: bool = False
    projection_bounds: tuple[float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class CachedPromptSteering:
    """A caller-fitted S-PSR module for the input of one decoder layer."""

    core: SteeringCore
    layer: int
    fitted: torch.nn.Module
    evidence_ref: str


def render_adaptive(session, native, *, steering, strength, max_new_tokens, **request):
    model = native.native_model
    layers = model.model.language_model.layers
    width = model.config.text_config.hidden_size
    if not isinstance(steering, (CachedGate, CachedPromptSteering)):
        raise TypeError("Provide CachedGate or CachedPromptSteering")
    if (not isinstance(steering.evidence_ref, str) or not steering.evidence_ref.strip()
            or type(steering.layer) is not int or not 0 <= steering.layer < len(layers)):
        raise ValueError("Provide calibration evidence and a native decoder layer index")
    layer = layers[steering.layer]
    core = steering.core
    require_steering_core(core)
    direction = fitted = gate = None
    if isinstance(steering, CachedGate):
        method, intervention = steering.method, steering.intervention
        if intervention not in ("residual", "raw_adapter"):
            raise ValueError("Select residual or raw_adapter intervention")
        if intervention == "raw_adapter":
            if not 0 <= strength <= 1:
                raise ValueError("Arete RAW adapter strength must be in [0,1]")
            if steering.layer != 0 or steering.preserve_norm or steering.projection_bounds is not None:
                raise ValueError("RAW adapter gating reads layer 0 before all adapter targets")
        if type(steering.preserve_norm) is not bool:
            raise ValueError("preserve_norm must be Boolean")
        if steering.projection_bounds is not None:
            low, high = steering.projection_bounds
            if not isfinite(low) or not isfinite(high) or low > high:
                raise ValueError("Projection bounds must be finite and ordered")
        if method != "random" or intervention == "residual":
            direction = steering.direction
            if (not isinstance(direction, torch.Tensor) or not direction.is_floating_point()
                    or direction.shape != (width,)):
                raise ValueError("Provide the calibrated direction at the selected decoder input")
            direction = direction.detach().float()
            length = direction.norm().item()
            if not isfinite(length) or length == 0:
                raise ValueError("The calibrated direction must be finite and nonzero")
            direction = direction / length
        gate = core.adaptive.ActivationGate(method, steering.parameter, seed=steering.seed,
                                           clabs=steering.clabs)
    else:
        method, intervention = "s_psr", "residual"
        if (not isinstance(steering.fitted, core.prompt.PromptSteering)
                or steering.fitted.probe.in_features != width):
            raise ValueError("Provide a fitted shared S-PSR module with the native decoder width")
        # Accelerate can move a layer's inputs inside its forward wrapper.
        # The pre-hook edits the actual input device before dispatch.
        fitted = deepcopy(steering.fitted).eval().requires_grad_(False)

    trace, applied = [], []
    receipt = dict(core_revision=CORE_REVISION, method=method, intervention=intervention,
                   decoder_input_layer=steering.layer, evidence_ref=steering.evidence_ref,
                   nominal_strength=strength, cached_steps=trace)
    if gate is not None:
        receipt.update(parameter=steering.parameter, seed=steering.seed,
                       clabs=asdict(steering.clabs) if steering.clabs is not None else None,
                       preserve_norm=steering.preserve_norm,
                       projection_bounds=steering.projection_bounds)

    prefix = cache = position = None
    scale = ExitStack()

    def before_forward(_model, _args, kwargs):
        nonlocal prefix, cache, position
        if position is not None:
            raise RuntimeError("The selected Arete decoder layer was not reached")
        ids, mask, past = (kwargs.get(key) for key in ("input_ids", "attention_mask", "past_key_values"))
        index = len(applied)
        if ids is None or mask is None or past is None or ids.ndim != 2:
            raise RuntimeError("Adaptive Arete requires native token IDs, a mask and a cache")
        if index == 0:
            prefix, cache = ids.shape[1], past
        expected_width = 1 if index else prefix
        past_length = prefix + index - expected_width
        if (index >= max_new_tokens or ids.shape != (1, expected_width)
                or mask.shape != (1, prefix + index) or past is not cache
                or past.get_seq_length() != past_length):
            raise RuntimeError("Arete adaptive steering saw an unexpected cached position")
        position = index

    def before_decoder(_layer, args):
        nonlocal position
        if position is None:
            raise RuntimeError("Arete visited the selected decoder layer more than once")
        index, position = position, None
        if index == 0 or strength == 0:
            applied.append(0.0)
            return None
        hidden = args[0]
        if hidden.shape != (1, 1, width):
            raise RuntimeError("Expected one full-width native Arete cached residual")
        record = dict(output_index=index, cache_position=prefix + index - 1)
        edited = hidden
        if fitted is not None:
            fitted.to(device=hidden.device)
            edited, coefficient = fitted(hidden, strength)
            effective = (coefficient * strength).item()
            if not isfinite(effective):
                raise ValueError("Nonfinite Arete S-PSR effective strength")
            record["coefficient"] = coefficient.item()
        else:
            score = None
            if method != "random":
                score = core.adaptive.activation_projection(hidden, direction, strength=strength).item()
            decision = gate.step(score)
            effective = strength if decision.fired else 0.0
            record.update(alignment=score, fired=decision.fired, probability=decision.probability)
            if intervention == "raw_adapter" and decision.fired:
                scale.enter_context(native.network.scaled(strength))
            if intervention == "residual":
                edited = core.adaptive.steer_residual(
                    hidden, direction, strength=strength, gate=decision.fired,
                    preserve_norm=steering.preserve_norm, projection_bounds=steering.projection_bounds,
                )
        if edited is not hidden and not torch.isfinite(edited).all().item():
            raise ValueError("Nonfinite Arete residual after steering")
        record["effective_strength"] = effective
        applied.append(effective)
        trace.append(record)
        return (edited, *args[1:]) if edited is not hidden else None

    def after_forward(_model, _args, output):
        scale.close()
        if output is not None and position is not None:
            raise RuntimeError("Arete finished a token without reaching the selected layer")

    with native.network.scaled(0.0), ExitStack() as hooks:
        hooks.callback(model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        hooks.callback(layer.register_forward_pre_hook(before_decoder).remove)
        hooks.callback(model.register_forward_hook(after_forward, always_call=True).remove)
        hooks.callback(scale.close)
        output = session.render(strength=0.0, max_new_tokens=max_new_tokens, **request)
    if not applied or len(applied) != len(output.native["new_token_ids"]):
        raise RuntimeError("Arete produced an unexpected number of adaptive tokens")
    receipt.update(prefix_tokens=prefix, effective_strengths=applied)
    if gate is not None:
        receipt["usage"] = asdict(gate.usage)
    output.native["adaptive_steering"] = receipt
    return output
