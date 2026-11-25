"""Frozen, answer-only A-PSR over the existing native Arete RAW render owner."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
from io import BytesIO
import json
from math import isfinite
import re
from struct import pack, unpack
from sys import float_info
from types import ModuleType, SimpleNamespace

import torch

from .steering_core import CORE_REVISION, DOSE_CORE_REVISION, require_dose_matcher, require_steering_core


_RAW_PROBE = "arete.authored_APSR.raw_probe_checkpoint.v1"
_LEAVES = ("direction.weight", "probe.weight", "probe.bias")
_LAYER_COUNT = 64
_WIDTH = 5120


@dataclass(frozen=True)
class _AllLayerBinding:
    probes: torch.nn.ModuleList
    layer_devices: tuple[torch.device, ...]
    base_file_sha256: str
    base_packet_sha256: str
    probe_sha256: str
    plan_sha256: str
    evidence_ref: str
    completed_updates: int


@dataclass(frozen=True)
class _ArchivedAllLayerSession:
    manifest: dict
    binding: object
    _open: object
    render: object
    selection = "raw"


@dataclass(frozen=True)
class _AllLayerConstant:
    case_ref: str
    eligible_tokens: int
    scope_sha256: str
    adaptive_trace_sha256: str
    strengths: tuple[float, ...]
    adaptive_doses: tuple[Fraction, ...]
    nearest_doses: tuple[Fraction, ...]
    matcher: ModuleType


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require_sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("An exact lowercase SHA-256 is required")
    return value


def _decode_bound(blob, ref):
    if (type(blob) is not bytes or type(ref.get("bytes")) is not int
            or len(blob) != ref["bytes"] or sha256(blob).hexdigest() != _require_sha(ref["sha256"])):
        raise ValueError("Artifact bytes differ from the authored checkpoint provenance")
    return blob


def load_all_layer_prompt_steering(
    core, *, probe_bytes, probe_sha256, training_plan_bytes, base_raw_bytes, evidence_ref
):
    """Load one exact authored RAW probe bank and the packet for its original native owner.

    The caller supplies independently verified, authorized bytes. Open the returned
    base packet through its original native backend. This helper owns no model.
    """
    require_steering_core(core)
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        raise ValueError("Provide evidence for this probe and its intended evaluation")
    if type(probe_bytes) is not bytes or sha256(probe_bytes).hexdigest() != _require_sha(probe_sha256):
        raise ValueError("Authored probe checksum differs")
    probe = torch.load(BytesIO(probe_bytes), weights_only=True, map_location="cpu")
    if (type(probe) is not dict
            or set(probe) != {"schema", "provenance", "cursor", "raw", "exact_full_first", "ema_decay"}
            or probe["schema"] != _RAW_PROBE or probe["ema_decay"] is not None):
        raise ValueError("Expected the authored RAW probe envelope with EMA disabled")
    completed = probe["cursor"]["completed"]
    if type(completed) is not int or completed < 1:
        raise ValueError("Expected a completed authored RAW update")

    source = probe["provenance"]
    plan_ref, base_ref = source["plan"], source["exact_RAW8"]
    plan = json.loads(_decode_bound(training_plan_bytes, plan_ref))
    if (not plan.get("schema", "").startswith("arete.authored_apsr.")
            or plan["raw"] != base_ref or plan["exact_no_update_data"] != source["measured_no_update"]
            or plan["ema_decay"] is not None):
        raise ValueError("The authored plan or recorded base identity differs")
    devices = tuple(torch.device(device) for device in plan["layer_devices"])
    if len(devices) != _LAYER_COUNT:
        raise ValueError("The authored Arete placement needs all 64 decoder blocks")

    base = torch.load(BytesIO(_decode_bound(base_raw_bytes, base_ref)), weights_only=True, map_location="cpu")
    if (type(base) is not dict or base.get("selection") != "raw"
            or base.get("origin", {}).get("completed_steps") != 8):
        raise ValueError("Expected the exact original native RAW8 packet")
    base_packet_sha = _require_sha(base["sha256"])

    saved = probe["raw"]
    layout = {
        f"{index}.{leaf}": (1,) if leaf == "probe.bias" else (1, _WIDTH)
        for index in range(_LAYER_COUNT)
        for leaf in _LEAVES
    }
    if not isinstance(saved, dict) or set(saved) != layout.keys():
        raise ValueError("Expected the ordered all-layer native probe state")
    for name, shape in layout.items():
        tensor = saved[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or tensor.shape != shape or tensor.layout != torch.strided
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("Malformed or nonfinite native FP32 probe tensor")

    probes = torch.nn.ModuleList([
        core.prompt.PromptSteering(_WIDTH, device="meta", dtype=torch.float32).to_empty(device="cpu")
        for _ in range(_LAYER_COUNT)
    ])
    probes.load_state_dict(saved, strict=True)
    probes.eval().requires_grad_(False)
    steering = _AllLayerBinding(probes, devices, base_ref["sha256"], base_packet_sha, probe_sha256,
                                plan_ref["sha256"], evidence_ref, completed)
    return steering, base


def bind_archived_all_layer_session(native, *, base_packet, check, render_context):
    """Bind the already verified RAW8 restore inside its original guarded cached owner.

    Pass the archive's existing scope check and native text device context factory.
    The caller retains the archive's resource and model lifetime.
    """
    from arete_native95.contracts import PROTOCOL
    from arete_native95.finished_text import generate
    from arete_native95.native import NativeResponse
    from particle_sliders.inference import _validate_packet, native_identity
    from particle_sliders.pg09 import _hash

    check()
    if not isinstance(native, NativeResponse) or native._native_closed:
        raise ValueError("Expected the open original archived Arete owner")
    _validate_packet(base_packet)
    identity = native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL)
    raw_state_sha256 = _hash(base_packet["network"])
    if (base_packet["origin"]["pg_version"] != "0.11.0" or base_packet["origin"]["completed_steps"] != 8
            or base_packet["specification"]["identity"] != identity
            or _hash(native.network.state_dict()) != raw_state_sha256):
        raise ValueError("The archive must already own the exact source-verified RAW8 state")

    def opened():
        check()
        if native._native_closed:
            raise RuntimeError("Archived Arete RAW session already closed")

    def render(**request):
        opened()
        if _hash(native.network.state_dict()) != raw_state_sha256:
            raise RuntimeError("The archived Arete RAW8 state changed before rendering")
        with render_context():
            output = generate(native, **request)
        if _hash(native.network.state_dict()) != raw_state_sha256:
            raise RuntimeError("The archived Arete RAW8 state changed during rendering")
        return output

    return _ArchivedAllLayerSession(
        manifest={"sha256": base_packet["sha256"]}, _open=opened, render=render,
        binding=SimpleNamespace(network=native.network, close=native.close, render=render),
    )


def _owned_native(session):

    session._open()
    binding = session.binding
    native = getattr(binding.close, "__self__", None)
    owner_package = type(native).__module__
    if owner_package == "arete_native95.native" and not isinstance(session, _ArchivedAllLayerSession):
        raise ValueError("Bind the already verified archived RAW8 restore first")
    if (owner_package not in {"arete_native.native", "arete_native95.native"}
            or not isinstance(native, import_module(owner_package).NativeResponse)
            or session.selection != "raw" or native.network is not binding.network or binding.render is None):
        raise ValueError("Expected an owned Arete native RAW render session")
    return native


def _effective_prefix(trace):
    scope = trace.get("prefix_dose_scope")
    if not isinstance(scope, dict):
        raise ValueError("Use a registered native answer prefix for dose matching")
    eligible, budget = scope.get("eligible_tokens"), scope.get("max_new_tokens")
    rows, prefix = trace.get("answer_steps"), trace.get("prefix_tokens")
    case_ref = scope.get("case_ref")
    if (not isinstance(case_ref, str) or not case_ref.strip()
            or type(eligible) is not int or type(budget) is not int or not 1 <= eligible <= budget <= 4096
            or type(prefix) is not int or prefix < 1 or scope.get("prompt_tokens") != prefix
            or not isinstance(rows, list) or not 1 <= len(rows) <= budget
            or trace.get("eligible_prefix_complete") is not (len(rows) >= eligible)):
        raise ValueError("Invalid native answer clock or prefix completion")
    _require_sha(scope.get("prompt_ids_sha256"))
    identities = ("base_raw_sha256", "base_packet_sha256", "authored_probe_sha256", "authored_plan_sha256")
    if any(_require_sha(scope.get(name)) != trace.get(name) for name in identities):
        raise ValueError("The recorded prefix and RAW provenance differ")
    prefix_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Invalid native per-layer gain trace")
        bits = row.get("effective_strength_float32_bits")
        if (row.get("output_index") != index or row.get("cache_position") != prefix + index - 1
                or not isinstance(bits, list) or len(bits) != _LAYER_COUNT
                or any(not isinstance(bits_value, str) or re.fullmatch(r"[a-f0-9]{8}", bits_value) is None
                       for bits_value in bits)):
            raise ValueError("Invalid native per-layer gain trace")
        values = tuple(unpack(">f", bytes.fromhex(bits_value))[0] for bits_value in bits)
        if not all(map(isfinite, values)) or (index >= eligible and any(values)):
            raise ValueError("Nonfinite native gain or nonzero edit outside the registered prefix")
        if index < eligible:
            prefix_rows.append(values)
    return tuple(zip(*prefix_rows, strict=True))


def match_all_layer_constant(adaptive_trace, *, matcher):
    """Build the nearest binary32 constant per native decoder direction.

    Pass ``output.native['adaptive_steering']`` from a completed registered prefix.
    Matching uses the signed effective gains on unit answer-position intervals.
    """
    require_dose_matcher(matcher)
    if (not isinstance(adaptive_trace, dict) or adaptive_trace.get("method") != "a_psr"
            or adaptive_trace.get("raw_only") is not True or adaptive_trace.get("ema_decay", False) is not None
            or adaptive_trace.get("decoder_output_layers") != list(range(_LAYER_COUNT))):
        raise ValueError("Expected a native RAW all-layer adaptive trace")
    columns = _effective_prefix(adaptive_trace)
    if not adaptive_trace["eligible_prefix_complete"]:
        raise ValueError("The adaptive answer ended before the registered prefix")
    scope = adaptive_trace["prefix_dose_scope"]
    coordinates = range(scope["eligible_tokens"] + 1)
    matches = tuple(matcher.match_binary32_interval_dose(column, coordinates) for column in columns)
    return _AllLayerConstant(scope["case_ref"], scope["eligible_tokens"], _digest(scope), _digest(adaptive_trace),
                             tuple(match.constant_strength for match in matches),
                             tuple(match.adaptive_dose for match in matches),
                             tuple(match.constant_dose for match in matches), matcher)


def _constant_dose_receipt(trace, constant):
    columns = _effective_prefix(trace)
    for column, value in zip(columns, constant.strengths, strict=True):
        if any(gain != value for gain in column):
            raise ValueError("The native control used a gain other than the matched constant")
    coordinates = range(len(columns[0]) + 1)
    observed = tuple(constant.matcher.interval_dose(column, coordinates, coefficient_format="binary32")
                     for column in columns)
    complete = trace["eligible_prefix_complete"]
    if complete and observed != constant.nearest_doses:
        raise RuntimeError("The observed constant dose differs from the matched control")
    exact = observed == constant.adaptive_doses if complete else None
    return dict(
        core_revision=DOSE_CORE_REVISION, adaptive_trace_sha256=constant.adaptive_trace_sha256,
        status=("exact" if exact else "nearest_binary32") if complete else "incomplete_prefix",
        exact_dose_equal=exact,
        layers=[dict(layer=index, adaptive_dose=str(adaptive), constant_dose=str(rendered),
                     signed_residual=str(rendered - adaptive), constant_float32_bits=pack(">f", value).hex())
                for index, (adaptive, rendered, value) in enumerate(zip(
                    constant.adaptive_doses, observed, constant.strengths, strict=True))],
    )


def generate_all_layer(session, *, steering, strength, max_new_tokens, eligible_tokens=None, case_ref=None, **request):
    """Apply the frozen authored A-PSR on every cached Arete decoder block.

    Strength is explicit. The returned trace records the actual per-layer FP32
    factors at each generated answer position. Opt in to a registered prefix with
    ``eligible_tokens`` and ``case_ref``. This function does no calibration.
    """
    if type(strength) not in (float, int) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("Provide an explicit finite A-PSR strength")
    return _generate_all_layer(session, steering=steering, strength=float(strength), max_new_tokens=max_new_tokens,
                               eligible_tokens=eligible_tokens, case_ref=case_ref, constant=None, request=request)


def generate_all_layer_constant(session, *, steering, constant, case_ref, max_new_tokens, **request):
    """Replay a matched constant over the same native answer prefix and RAW owner."""
    if not isinstance(constant, _AllLayerConstant) or case_ref != constant.case_ref:
        raise ValueError("Match the constant on this exact registered case first")
    return _generate_all_layer(session, steering=steering, strength=None, max_new_tokens=max_new_tokens,
                               eligible_tokens=constant.eligible_tokens, case_ref=case_ref,
                               constant=constant, request=request)


def _all_layer_context(session, steering):
    if not isinstance(steering, _AllLayerBinding):
        raise ValueError("Load the exact authored A-PSR and base artifacts first")
    native = _owned_native(session)
    if session.manifest.get("sha256") != steering.base_packet_sha256:
        raise ValueError("This owner does not contain the exact original RAW8 packet")
    model = native.native_model
    layers = model.model.language_model.layers
    if (model.training or len(layers) != _LAYER_COUNT
            or model.config.text_config.hidden_size != _WIDTH
            or any(parameter.requires_grad for parameter in model.parameters())):
        raise ValueError("Expected the frozen original complete Arete decoder")

    placement = native.runtime.placement
    if placement["kind"] == "single":
        actual = (torch.device(placement["device"]),) * _LAYER_COUNT
    else:
        actual = tuple(torch.device(placement["device_map"][f"model.language_model.layers.{i}"])
                       for i in range(_LAYER_COUNT))
    if actual != steering.layer_devices:
        raise ValueError("Native layer placement differs from the exact authored plan")

    artifacts = dict(base_raw_sha256=steering.base_file_sha256, base_packet_sha256=steering.base_packet_sha256,
                     authored_probe_sha256=steering.probe_sha256, authored_plan_sha256=steering.plan_sha256,
                     completed_authored_updates=steering.completed_updates)
    return native, layers, actual, artifacts


def _prefix_dose_scope(native, actual, artifacts, *, case_ref, eligible_tokens, max_new_tokens, max_tokens, seed):
    return dict(case_ref=case_ref, eligible_tokens=eligible_tokens, max_new_tokens=max_new_tokens,
                max_tokens=max_tokens, seed=seed, owner=type(native).__module__,
                native_model_sha256=_digest(native.model_identity),
                generation_config_sha256=_digest(native.native_model.generation_config.to_dict()),
                layer_devices=[str(device) for device in actual], **artifacts)


def _generate_all_layer(session, *, steering, strength, max_new_tokens, eligible_tokens, case_ref, constant, request):
    if request.get("enable_thinking", False) is not False:
        raise ValueError("Authored answer-only A-PSR requires native thinking disabled")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Require the original native token-budget domain")
    if ((eligible_tokens is None and case_ref is not None)
            or (eligible_tokens is not None and (type(eligible_tokens) is not int
                or not 1 <= eligible_tokens <= max_new_tokens or not isinstance(case_ref, str) or not case_ref.strip()))):
        raise ValueError("Provide a finite registered answer prefix and nonempty case reference")
    native, layers, actual, artifacts = _all_layer_context(session, steering)
    model = native.native_model
    scope = None
    if eligible_tokens is not None:
        scope = _prefix_dose_scope(native, actual, artifacts, case_ref=case_ref, eligible_tokens=eligible_tokens,
                                   max_new_tokens=max_new_tokens, max_tokens=request.get("max_tokens", 4096),
                                   seed=request.get("seed", 0))
    probes = deepcopy(steering.probes) if strength or (constant is not None and any(constant.strengths)) else None
    answer_steps = []
    prefix = cache = None
    visited = None

    def before_forward(_model, _args, kw):
        nonlocal prefix, cache, visited
        if visited is not None:
            raise RuntimeError("Arete started a new cached step before closing all block visits")
        token_index = len(answer_steps)
        ids, mask, past = (kw.get(key) for key in ("input_ids", "attention_mask", "past_key_values"))
        if ids is None or ids.ndim != 2 or mask is None or past is None or kw.get("use_cache") is not True:
            raise RuntimeError("A-PSR requires the complete native cache and attention mask")
        if token_index == 0:
            prefix, cache = ids.shape[1], past
        count = prefix if token_index == 0 else 1
        if (prefix < 1 or token_index >= max_new_tokens or ids.shape != (1, count)
                or mask.shape != (1, prefix + token_index) or past is not cache
                or past.get_seq_length() != prefix + token_index - count
                or (token_index == 0 and not bool((mask == 1).all()))):
            raise RuntimeError("Unexpected native answer or cache position")
        if token_index == 0 and scope is not None:
            scope["prompt_ids_sha256"] = _digest(ids[0].tolist())
            scope["prompt_tokens"] = prefix
            if constant is not None and constant.scope_sha256 != _digest(scope):
                raise ValueError("Constant control does not match the native case, prompt, RAW, or clock")
        visited = []

    def edit_block(index):
        def after_block(_layer, _args, hidden):
            token_index = len(answer_steps)
            positions = prefix if token_index == 0 else 1
            if (visited is None or len(visited) != index or not isinstance(hidden, torch.Tensor)
                    or hidden.shape != (1, positions, _WIDTH) or hidden.dtype != torch.bfloat16
                    or hidden.device != actual[index]):
                raise RuntimeError("Unexpected Arete BF16 post-block visit or device placement")
            if probes is None or (eligible_tokens is not None and token_index >= eligible_tokens):
                visited.append(None)
                return None

            residual = hidden[:, -1:, :] if token_index == 0 else hidden
            if constant is None:
                probe = probes[index].to(device=residual.device)
                edited, coefficient = probe(residual, strength)
                # This exactly mirrors the FP32 product used by shared PromptSteering.forward.
                effective = coefficient * coefficient.new_tensor(strength)
                if not bool(torch.isfinite(effective).all()):
                    raise ValueError("An authored layer produced a nonfinite coefficient")
                visited.append((coefficient.item(), effective.item()))
            else:
                effective = constant.strengths[index]
                if effective == 0:
                    visited.append(None)
                    return None
                direction = probes[index].direction.weight[0].to(device=residual.device)
                edited = (residual.float() + effective * direction).to(residual.dtype)
                visited.append((None, effective))
            if not bool(torch.isfinite(edited).all()):
                raise ValueError("An authored layer produced a nonfinite residual")
            if token_index == 0 and prefix > 1:
                return torch.cat((hidden[:, :-1, :], edited), dim=1)
            return edited
        return after_block

    def after_forward(_model, _args, output):
        nonlocal visited
        if output is None:
            return
        if visited is None or len(visited) != _LAYER_COUNT:
            raise RuntimeError("A cached native step omitted an authored layer")
        token_index = len(answer_steps)
        if output.past_key_values is not cache or cache.get_seq_length() != prefix + token_index:
            raise RuntimeError("A native answer step replaced or misadvanced the cache")
        answer_steps.append(dict(
            output_index=token_index, cache_position=prefix + token_index - 1,
            coefficient_float32_bits=[None if pair is None or pair[0] is None else pack(">f", pair[0]).hex()
                                      for pair in visited],
            effective_strength_float32_bits=[
                "00000000" if pair is None else pack(">f", pair[1]).hex() for pair in visited],
        ))
        visited = None

    with native.network.scaled(0.0), ExitStack() as hooks:
        hooks.callback(model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        for index, layer in enumerate(layers):
            hooks.callback(layer.register_forward_hook(edit_block(index)).remove)
        hooks.callback(model.register_forward_hook(after_forward).remove)
        # The original owner retains all native generation, RNG and adapter lifecycle handling.
        output = session.render(strength=0.0, max_new_tokens=max_new_tokens, **request)
    if (not answer_steps or len(answer_steps) != len(output.native["new_token_ids"])
            or output.native["prompt_tokens"] != prefix
            or output.native["selection"] != "raw" or output.native["weight_EMA"] is not None
            or output.native["enable_thinking"] is not False):
        raise RuntimeError("The A-PSR trace does not bind each native answer token")
    trace = dict(
        core_revision=CORE_REVISION, method="a_psr" if constant is None else "a_psr_constant",
        intervention="residual", position_scope="answer_only",
        decoder_output_layers=list(range(_LAYER_COUNT)), nominal_strength=strength,
        prefix_tokens=prefix, answer_steps=answer_steps, raw_only=True, ema_decay=None,
        **artifacts, evidence_ref=steering.evidence_ref,
    )
    if scope is not None:
        native_policy = output.native.get("explicit_generation_config", output.native.get("actual_generate_kwargs"))
        if (output.native.get("seed") != scope["seed"] or not isinstance(native_policy, dict)
                or native_policy.get("max_new_tokens") != max_new_tokens):
            raise RuntimeError("The finished text does not bind the registered native seed and token budget")
        complete = len(answer_steps) >= eligible_tokens
        trace.update(prefix_dose_scope=scope, eligible_prefix_complete=complete)
        if not complete:
            trace["incomplete_prefix_reason"] = "The native answer ended before the registered prefix"
    if constant is not None:
        trace["dose_comparison"] = _constant_dose_receipt(trace, constant)
    output.native["adaptive_steering"] = trace
    return output
