"""Durable firsts for preregistered native Arete A-PSR prefix arms."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from .all_layer import (
    _all_layer_context, _constant_dose_receipt, _digest, _prefix_dose_scope,
    generate_all_layer, generate_all_layer_constant, match_all_layer_constant,
)
from .observer_ingress import _ARMS, _INTENT, _STAT_FIELDS, _native_trace, _read_registration, _registered_first
from .steering_core import require_dose_matcher


def _arm_paths(registration, arm):
    registered = registration["arms"][arm]
    paths = {key: Path(registered[f"{key}_path"]) for key in ("claim", "text", "native", "first")}
    for key, path in paths.items():
        if (not path.is_absolute() or str(path) != registered[f"{key}_path"]
                or path.parent.resolve(strict=True) != path.parent or not path.parent.is_dir()
                or path.name in {".", ".."}):
            raise ValueError("Expected each first artifact at its exact canonical path")
    if paths["native"] != Path(str(paths["text"]) + ".native.json"):
        raise ValueError("Register the original native text writer's sidecar path")
    return paths


def _sync_parent(path):
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_receipt(path, record):
    data = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_parent(path)
    return dict(path=str(path), bytes=len(data), sha256=sha256(data).hexdigest())


def _file_ref(path):
    digest = sha256()
    size = 0
    initial = path.lstat()

    def check(info):
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or any(getattr(initial, field) != getattr(info, field) for field in _STAT_FIELDS)):
            raise ValueError("First artifact changed or is not an independent regular file")

    check(initial)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        check(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        check(os.fstat(stream.fileno()))
    check(path.lstat())
    if size != initial.st_size:
        raise ValueError("First artifact changed during read")
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def _settled_first(registration_ref, registration, registered_at, arm, paths):
    if not os.path.lexists(paths["first"]):
        if any(os.path.lexists(path) for path in paths.values()):
            raise RuntimeError(f"The registered {arm} arm already entered or has a partial first")
        return None, None
    reference = _file_ref(paths["first"])
    _, native = _registered_first(registration_ref, registration, arm, reference, registered_at)
    trace = _native_trace(native, registration["scope"], arm, registration["adaptive_strength"])
    return reference, trace


def _registered_request(session, steering, scope, ids):
    native, _, devices, artifacts = _all_layer_context(session, steering)
    config = native.native_model.config.text_config
    seed, max_tokens = scope["seed"], scope["max_tokens"]
    budget = scope["max_new_tokens"]
    if (not isinstance(ids, (list, tuple)) or type(seed) is not int or not 0 <= seed < 2**63 - 100
            or type(max_tokens) is not int or not 1 <= max_tokens <= 32768
            or not 1 <= len(ids) <= max_tokens or len(ids) + budget > config.max_position_embeddings
            or any(type(token) is not int or not 0 <= token < config.vocab_size for token in ids)):
        raise ValueError("Expected the registered complete native input IDs and seed")
    ids = list(ids)
    clock = {key: scope[key] for key in ("case_ref", "eligible_tokens", "max_new_tokens", "max_tokens", "seed")}
    actual = _prefix_dose_scope(native, devices, artifacts, **clock)
    actual.update(prompt_ids_sha256=_digest(ids), prompt_tokens=len(ids))
    if actual != scope:
        raise ValueError("The native case, prompt IDs, clock, or RAW binding differs from registration")
    return dict(clock, ids=ids, enable_thinking=False)


def render_prefix_first(registration_ref, arm, session, *, steering, ids, matcher):
    """Render one registered arm through its caller-owned, serialized native RAW owner.

    Returns the exact first reference, including when the arm was already settled.
    An adaptive first ending before the registered prefix leaves constant
    ineligible and returns None for that arm. Entered and partial arms are
    retained and raise on subsequent attempts. The caller controls resource,
    role and input access through the existing native owner and its guard.
    """
    require_dose_matcher(matcher)
    if arm not in _ARMS:
        raise ValueError("Expected one registered OFF, adaptive or constant arm")
    registration, registered_at = _read_registration(registration_ref)
    scope = registration["scope"]
    paths = _arm_paths(registration, arm)
    existing, existing_trace = _settled_first(registration_ref, registration, registered_at, arm, paths)

    constant = None
    if arm == "constant":
        _, adaptive = _settled_first(registration_ref, registration, registered_at, "adaptive",
                                     _arm_paths(registration, "adaptive"))
        if adaptive is None:
            raise ValueError("The matched constant requires its settled adaptive first")
        if not adaptive["eligible_prefix_complete"]:
            if existing is not None:
                raise ValueError("A settled constant cannot follow an incomplete adaptive prefix")
            return None
        constant = match_all_layer_constant(adaptive, matcher=matcher)
    if existing is not None:
        if constant is not None and existing_trace.get("dose_comparison") != _constant_dose_receipt(existing_trace, constant):
            raise ValueError("Settled native constant differs from the computed matched dose")
        return existing

    request = _registered_request(session, steering, scope, ids)
    intent = dict(schema=_INTENT, registration=registration_ref, arm=arm, case_ref=scope["case_ref"], ema_decay=None)
    claim = _write_receipt(paths["claim"], intent)
    if arm == "constant":
        del request["eligible_tokens"]
        output = generate_all_layer_constant(session, steering=steering, constant=constant, **request)
    else:
        strength = 0.0 if arm == "off" else registration["adaptive_strength"]
        output = generate_all_layer(session, steering=steering, strength=strength, **request)

    # Preserve the original returned output even if its trace cannot settle a first.
    output.save(paths["text"])
    _sync_parent(paths["text"])
    trace = _native_trace(output.native, scope, arm, registration["adaptive_strength"])
    if constant is not None and trace.get("dose_comparison") != _constant_dose_receipt(trace, constant):
        raise ValueError("Finished native constant differs from the computed matched dose")
    first = dict(claim=claim, text=_file_ref(paths["text"]), native=_file_ref(paths["native"]))
    return _write_receipt(paths["first"], first)
