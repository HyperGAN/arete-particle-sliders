"""Join registered native A-PSR firsts for the existing Arete observers."""

import json
from math import isfinite
import os
from pathlib import Path
import re
import stat

from .all_layer import _constant_dose_receipt, _decode_bound, _effective_prefix, match_all_layer_constant
from .steering_core import CORE_REVISION, require_dose_matcher


_REGISTRATION = "arete.a_psr.prefix_observer_registration.v1"
_INTENT = "arete.a_psr.prefix_native_intent.v1"
_ARMS = ("off", "adaptive", "constant")
_NATIVE = {"arete095-native-finished-text-v1", "arete-native-finished-text-v2", "arete-native-native-finished-text-v1"}
# References from the existing adopted Arete v3 readout plan. Its observer guard
# checks these files in its own aperture, including its pinned integrity qualification.
_V3_OBSERVERS = {
    "calibration_plan": (14613, "148fe16a1e55487e5359121ca58606c269cab77021227798810893ad71d6bea6"),
    "tone_qualification": (7415, "be4162616893fed5490e9b67d1c65f12f854d97d450e99f7ef98996d4fb8d92a"),
    "guarded_calibration": (225, "c153252738897033d3a748585d9c12d3a69a8d8534ec524e3f69e41b7048d6ca"),
}
_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _pinned_stat(ref):
    if type(ref) is not dict or set(ref) != {"path", "bytes", "sha256"}:
        raise ValueError("Expected one exact first reference")
    path = Path(ref["path"])
    original = path.lstat()
    if (str(path) != ref["path"] or path.resolve(strict=True) != path or not stat.S_ISREG(original.st_mode)
            or original.st_nlink != 1):
        raise ValueError("Expected a regular first at its exact absolute path")
    if original.st_size != ref["bytes"]:
        raise ValueError("First size differs from the frozen reference")
    return path, original


def _verified_bytes(ref, *, not_before=None, not_after=None):
    path, original = _pinned_stat(ref)

    def check(info):
        if ((not_before is not None and info.st_mtime_ns < not_before)
                or (not_after is not None and info.st_mtime_ns > not_after)):
            raise ValueError("An output or first disagrees with the chronology of its registered native claim")
        # Reading may change access time, but must preserve content metadata.
        if any(getattr(original, key) != getattr(info, key) for key in _STAT_FIELDS):
            raise ValueError("First changed while being verified")

    check(original)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        check(os.fstat(stream.fileno()))
        data = _decode_bound(stream.read(original.st_size + 1), ref)
        check(os.fstat(stream.fileno()))
    check(path.lstat())
    return data, original.st_mtime_ns


def _verified_json(ref, **chronology):
    def unique(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("Duplicate key in a frozen claim or first")
        return value

    data, modified = _verified_bytes(ref, **chronology)
    return json.loads(data, object_pairs_hook=unique), modified


def _qualified_observers(binding):
    if type(binding) is not dict or binding.keys() != _V3_OBSERVERS.keys():
        raise ValueError("Expected the existing qualified observer references")
    for name, (size, digest) in _V3_OBSERVERS.items():
        ref = binding[name]
        if type(ref) is not dict or ref.get("bytes") != size or ref.get("sha256") != digest:
            raise ValueError("The exact guarded v3 tone and separate complete-text integrity qualifications are required")
        _pinned_stat(ref)


def _registered_first(registration_ref, registration, arm, first_ref, registered_at):
    expected = registration["arms"][arm]
    if type(first_ref) is not dict or first_ref.get("path") != expected["first_path"]:
        raise ValueError("First path differs from the exact registered arm")
    first, first_time = _verified_json(first_ref, not_before=registered_at)
    if (type(first) is not dict or any(type(first.get(name)) is not dict
            or first[name].get("path") != expected[f"{name}_path"] for name in ("claim", "text", "native"))):
        raise ValueError("First does not identify the exact registered arm and claim")
    intent, intent_time = _verified_json(first["claim"], not_before=registered_at, not_after=first_time)
    expected_intent = dict(schema=_INTENT, registration=registration_ref, arm=arm,
                           case_ref=registration["scope"]["case_ref"], ema_decay=None)
    if type(intent) is not dict or any(key not in intent or intent[key] != value for key, value in expected_intent.items()):
        raise ValueError("Native claim does not bind the exact registration and arm")
    native, _ = _verified_json(first["native"], not_before=intent_time, not_after=first_time)
    raw_text, _ = _verified_bytes(first["text"], not_before=intent_time, not_after=first_time)
    # Both existing observers require original UTF-8. Keep the original reference for each.
    raw_text.decode("utf-8")
    return first, native


def _native_trace(native, scope, arm, adaptive_strength):
    trace = native["adaptive_steering"]
    budget = scope["max_new_tokens"]
    tokens = native["new_token_ids"]
    eos = native["native_eos_token_id"]
    eos = [eos] if type(eos) is int else eos
    policy = native.get("explicit_generation_config", native.get("actual_generate_kwargs"))
    expected_method = "a_psr_constant" if arm == "constant" else "a_psr"
    expected_strength = {"off": 0.0, "adaptive": adaptive_strength, "constant": None}[arm]
    actual_strength = trace["nominal_strength"]
    if (native["schema"] not in _NATIVE or native["selection"] != "raw" or native["weight_EMA"] is not None
            or type(native["strength"]) not in (int, float) or native["strength"] != 0
            or native["enable_thinking"] is not False or native["seed"] != scope["seed"]
            or native["prompt_tokens"] != scope["prompt_tokens"] or not isinstance(policy, dict)
            or policy.get("max_new_tokens") != budget or not isinstance(tokens, list)
            or not 1 <= len(tokens) <= budget or any(type(token) is not int or token < 0 for token in tokens)
            or not isinstance(eos, list) or not eos or any(type(token) is not int or token < 0 for token in eos)
            or any(token in eos for token in tokens[:-1]) or (len(tokens) < budget and tokens[-1] not in eos)
            or trace["prefix_dose_scope"] != scope or trace["core_revision"] != CORE_REVISION
            or trace["method"] != expected_method or trace["position_scope"] != "answer_only"
            or trace["intervention"] != "residual" or trace["raw_only"] is not True or trace["ema_decay"] is not None
            or trace["decoder_output_layers"] != list(range(64)) or actual_strength != expected_strength
            or (arm != "constant" and type(actual_strength) not in (int, float))
            or len(trace["answer_steps"]) != len(tokens)):
        raise ValueError("Finished native first disagrees with the registered RAW arm or answer clock")
    columns = _effective_prefix(trace)
    if arm == "off" and any(any(column) for column in columns):
        raise ValueError("The matched OFF first contains a native edit")
    return trace


def _timing(traces, matcher, dose_policy):
    adaptive, constant = traces.get("adaptive"), traces.get("constant")
    dose = None
    if constant is not None:
        if adaptive is None or not adaptive["eligible_prefix_complete"]:
            raise ValueError("A constant first requires its complete adaptive prefix first")
        plan = match_all_layer_constant(adaptive, matcher=matcher)
        dose = _constant_dose_receipt(constant, plan)
        if constant.get("dose_comparison") != dose:
            raise ValueError("Stored constant receipt differs from the exact native gains and rounding")
    reason = None
    if adaptive is not None and not adaptive["eligible_prefix_complete"]:
        reason = "adaptive_prefix_incomplete"
    elif dose is not None and dose["status"] == "incomplete_prefix":
        reason = "constant_prefix_incomplete"
    elif set(traces) != set(_ARMS):
        reason = "missing_registered_first"
    elif dose_policy == "exact" and not dose["exact_dose_equal"]:
        reason = "nearest_binary32_residual_under_exact_only_policy"
    return dict(status="unmeasured" if reason else "ready_for_observation", reason=reason, dose=dose)


def _read_registration(registration_ref):
    registration, registered_at = _verified_json(registration_ref)
    if (registration["schema"] != _REGISTRATION or registration["ema_decay"] is not None
            or set(registration["arms"]) != set(_ARMS)
            or registration["dose_policy"] not in {"exact", "nearest_binary32"}
            or type(registration["adaptive_strength"]) not in (int, float) or not isfinite(registration["adaptive_strength"])
            or not isinstance(registration["prompt"], str) or not registration["prompt"].strip()
            or not registration["integrity_output_contract"]):
        raise ValueError("Expected an exact preregistered RAW trio with EMA disabled")
    scope = registration["scope"]
    if (not isinstance(scope["case_ref"], str) or not scope["case_ref"].strip()
            or type(scope["eligible_tokens"]) is not int or type(scope["max_new_tokens"]) is not int
            or not 1 <= scope["eligible_tokens"] <= scope["max_new_tokens"] <= 4096):
        raise ValueError("Expected a registered native answer prefix")
    arms = registration["arms"]
    blind_ids = [arms[arm]["blind_id"] for arm in _ARMS]
    paths = [arms[arm][key] for arm in _ARMS for key in ("claim_path", "first_path", "text_path", "native_path")]
    if len(set(paths)) != len(paths) or len(set(blind_ids)) != len(_ARMS) or any(
        not isinstance(blind_id, str) or not re.fullmatch(r"tone-[0-9a-f]{12,64}", blind_id) for blind_id in blind_ids
    ):
        raise ValueError("Registered arms require distinct paths and opaque tone IDs")
    return registration, registered_at


def prepare_blinded_trio(registration_ref, first_refs, *, matcher):
    """Project frozen Arete native firsts into v3 tone and independent integrity rows.

    The caller supplies the independently registered case, exact first references,
    and archived qualified observer bindings. Absent firsts stay unmeasured. This
    verifies and reads those references and creates no files or observer calls.
    """
    require_dose_matcher(matcher)
    if not isinstance(first_refs, dict) or not set(first_refs) <= set(_ARMS):
        raise ValueError("Expected only registered native arm firsts")
    registration, registered_at = _read_registration(registration_ref)
    arms = registration["arms"]
    scope = registration["scope"]
    _qualified_observers(registration["observer_binding"])

    traces, tone_rows, integrity_rows, private = {}, [], [], {}
    native_identity = None
    for arm in sorted(first_refs, key=lambda key: arms[key]["blind_id"]):
        first_ref = first_refs[arm]
        first, native = _registered_first(registration_ref, registration, arm, first_ref, registered_at)
        file = first["text"]
        identity = tuple(native.get(key) for key in (
            "schema", "model", "placement", "native_eos_token_id", "native_pad_token_id", "native_bos_token_id",
            "explicit_generation_config", "actual_generate_kwargs",
        ))
        if native_identity is None:
            native_identity = identity
        elif identity != native_identity:
            raise ValueError("The native model, placement, or termination policy changed across arms")
        traces[arm] = _native_trace(native, scope, arm, registration["adaptive_strength"])
        blind = arms[arm]["blind_id"]
        row = private[arm] = dict(blind_id=blind, first=first_ref, claim=first["claim"], text=file, native=first["native"],
                                  tone_input_index=None, integrity_input_index=None)
        if file["bytes"] <= 16384:
            row["tone_input_index"] = len(tone_rows)
            tone_rows.append(dict(id=blind, prompt=registration["prompt"], output=file))
        if 0 < file["bytes"] <= 131072:
            row["integrity_input_index"] = len(integrity_rows)
            integrity_rows.append(dict(file=file, media_type="text", output_contract=registration["integrity_output_contract"]))

    timing = _timing(traces, matcher, registration["dose_policy"])
    if timing["status"] == "ready_for_observation" and not len(tone_rows) == len(integrity_rows) == len(_ARMS):
        timing.update(status="unmeasured", reason="observer_input_outside_existing_limits")
    return dict(
        tone_bank=dict(rows=tone_rows), integrity_inputs=integrity_rows, observer_binding=registration["observer_binding"],
        private=dict(registration=registration_ref, case_ref=scope["case_ref"], arms=private,
                     missing_firsts=[arm for arm in _ARMS if arm not in private], timing=timing),
    )
