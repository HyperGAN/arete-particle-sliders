"""Frozen, answer-only A-PSR over the existing native Arete RAW render owner."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from io import BytesIO
import json
import re
from struct import pack
from sys import float_info
from types import SimpleNamespace

import torch

from .steering_core import CORE_REVISION, require_steering_core


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


def generate_all_layer(session, *, steering, strength, max_new_tokens, **request):
    """Apply the frozen authored A-PSR on every cached Arete decoder block.

    Strength is explicit. The returned trace records the actual per-layer FP32
    factors at each generated answer position. This function does no calibration.
    """
    if not isinstance(steering, _AllLayerBinding):
        raise ValueError("Load the exact authored A-PSR and base artifacts first")
    if type(strength) not in (float, int) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("Provide an explicit finite A-PSR strength")
    strength = float(strength)
    if request.get("enable_thinking", False) is not False:
        raise ValueError("Authored answer-only A-PSR requires native thinking disabled")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Require the original native token-budget domain")
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

    probes = deepcopy(steering.probes) if strength else None
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
        visited = []

    def edit_block(index):
        def after_block(_layer, _args, hidden):
            token_index = len(answer_steps)
            positions = prefix if token_index == 0 else 1
            if (visited is None or len(visited) != index or not isinstance(hidden, torch.Tensor)
                    or hidden.shape != (1, positions, _WIDTH) or hidden.dtype != torch.bfloat16
                    or hidden.device != actual[index]):
                raise RuntimeError("Unexpected Arete BF16 post-block visit or device placement")
            if strength == 0:
                visited.append(None)
                return None

            residual = hidden[:, -1:, :] if token_index == 0 else hidden
            probe = probes[index].to(device=residual.device)
            edited, coefficient = probe(residual, strength)
            # This exactly mirrors the FP32 product used by shared PromptSteering.forward.
            effective = coefficient * coefficient.new_tensor(strength)
            if not bool(torch.isfinite(effective).all() and torch.isfinite(edited).all()):
                raise ValueError("An authored layer produced a nonfinite coefficient or residual")
            visited.append((coefficient.item(), effective.item()))
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
            coefficient_float32_bits=[None if pair is None else pack(">f", pair[0]).hex() for pair in visited],
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
    output.native["adaptive_steering"] = dict(
        core_revision=CORE_REVISION, method="a_psr", intervention="residual", position_scope="answer_only",
        decoder_output_layers=list(range(_LAYER_COUNT)), nominal_strength=strength,
        prefix_tokens=prefix, answer_steps=answer_steps, raw_only=True, ema_decay=None,
        base_raw_sha256=steering.base_file_sha256, base_packet_sha256=steering.base_packet_sha256,
        authored_probe_sha256=steering.probe_sha256, authored_plan_sha256=steering.plan_sha256,
        completed_authored_updates=steering.completed_updates, evidence_ref=steering.evidence_ref,
    )
    return output
