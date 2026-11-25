"""CPU fixtures for the full authored A-PSR format and cached answer-only path."""

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from importlib.util import find_spec
import json
import os
from pathlib import Path
from struct import unpack
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch
from transformers import DynamicCache, Qwen3_5TextConfig

from arete_native_scheduling import (
    bind_archived_all_layer_session, generate_all_layer, load_all_layer_prompt_steering, load_steering_core,
)


def packet(obj):
    b = BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def reference(blob):
    return dict(bytes=len(blob), sha256=sha256(blob).hexdigest())


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(64))


class NativeTrace(torch.nn.Module):
    omit_last = False

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(language_model=Decoder())
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=5120))
        self.trace = []

    def forward(self, *, input_ids, attention_mask, past_key_values, use_cache, position_ids):
        value = torch.zeros((*input_ids.shape, self.config.text_config.hidden_size), dtype=torch.bfloat16)
        value[:, :, 1] = input_ids
        layers = self.model.language_model.layers
        for layer in layers[:-1] if self.omit_last else layers:
            value = layer(value)
        keys = value[:, None, :, :1]
        past_key_values.update(keys, keys, 3)
        self.trace.append(value.clone())
        return SimpleNamespace(last_hidden_state=value, past_key_values=past_key_values)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("raw", torch.zeros(1))
        self.projections = [SimpleNamespace(multiplier=.75)]
        self.closed = False

    @property
    def scale(self):
        return self.projections[0].multiplier

    @contextmanager
    def scaled(self, value):
        projection = self.projections[0]
        original, projection.multiplier = projection.multiplier, value
        try:
            yield
        finally:
            projection.multiplier = original


class NativeOwner:
    __module__ = "arete_native.native"
    _native_closed = False
    model_identity = {"fixture": "synthetic"}

    def __init__(self):
        self.native_model = NativeTrace().eval()
        self.runtime = SimpleNamespace(placement={"kind": "single", "device": "cpu"})
        self.network = Network()

    def close(self):
        pass


class ArchivedNativeOwner(NativeOwner):
    __module__ = "arete_native95.native"


class Session:
    selection = "raw"

    def __init__(self, owner, identity, *, eos_after=None):
        self.manifest = {"sha256": identity}
        self.binding = SimpleNamespace(network=owner.network, close=owner.close, render=self.render)
        self.owner = owner
        self.eos_after = eos_after

    def _open(self):
        pass

    def render(self, *, strength, max_new_tokens, **_request):
        assert strength == 0
        model = self.owner.native_model
        config = Qwen3_5TextConfig(num_hidden_layers=64,
                                   layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16)
        cache = DynamicCache(config=config)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        tokens = []
        with self.owner.network.scaled(strength):
            assert self.owner.network.scale == 0
            for idx in range(max_new_tokens):
                output = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True,
                               position_ids=torch.arange(mask.numel() - ids.numel(), mask.numel())[None, :])
                cache = output.past_key_values
                mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
                token = 7 + int(output.last_hidden_state[0, -1, 0] > 0)
                if self.eos_after == idx + 1:
                    token = 2
                tokens.append(token)
                if token == 2:
                    break
                ids = torch.tensor([[token]])
        return SimpleNamespace(text=str(tokens), native=dict(
            new_token_ids=tokens, prompt_tokens=3, selection="raw", weight_EMA=None,
            enable_thinking=False, native_eos_token_id=2,
        ))


def observed(bits):
    return unpack(">f", bytes.fromhex(bits))[0]


class AllLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Official variant test supplies the shared wheel or constructs the identical
        # source archive from the package in the shared rollout worktree.
        cls._temp = TemporaryDirectory()
        cls.addClassCleanup(cls._temp.cleanup)
        wheel = os.getenv("ARETE_STEERING_CORE_WHEEL")
        if wheel is None:
            source = os.getenv("ARETE_STEERING_CORE_SOURCE")
            if source is None:
                raise unittest.SkipTest("Set ARETE_STEERING_CORE_WHEEL to the shared b1f780d wheel")
            root = Path(source)
            wheel = Path(cls._temp.name) / "shared-steering.whl"
            with ZipFile(wheel, "w") as out:
                for name in ("adaptive_steering.py", "prompt_steering.py"):
                    out.write(root / name, f"particle_sliders/{name}")
        cls.core = load_steering_core(wheel)

    def owned_binding(self, *, ema=None, invalid=None, devices=None, base=None):
        if base is None:
            base = dict(selection="raw", origin={"completed_steps": 8}, sha256="a" * 64)
        base = packet(base)
        base_ref = reference(base)
        measured = {"terminal": {"sha256": "e" * 64}}
        plan = json.dumps(dict(
            schema="arete.authored_apsr.exact_half_resume.plan.v1", raw=base_ref, exact_no_update_data=measured,
            layer_devices=devices or ["cpu"] * 64, ema_decay=None,
        )).encode()
        state = {}
        for layer in range(64):
            state[f"{layer}.direction.weight"] = torch.zeros((1, 5120))
            state[f"{layer}.probe.weight"] = torch.zeros((1, 5120))
            state[f"{layer}.probe.bias"] = torch.zeros(1)
        state["0.probe.bias"][0] = .25
        state["0.direction.weight"][0, 0] = 1
        state["1.probe.weight"][0, 0] = 1
        state["1.direction.weight"][0, 0] = 1
        if invalid:
            state[invalid][0] = float("nan")
        raw = packet(dict(
            schema="arete.authored_APSR.raw_probe_checkpoint.v1", cursor={"completed": 15},
            raw=state, exact_full_first={"sha256": "b" * 64}, ema_decay=ema,
            provenance=dict(plan=reference(plan), exact_RAW8=base_ref, measured_no_update=measured),
        ))
        return dict(core=self.core, probe_bytes=raw, probe_sha256=reference(raw)["sha256"],
                    training_plan_bytes=plan, base_raw_bytes=base, evidence_ref="synthetic fixture")

    def test_full_probe_schema_rejects_wrong_source_base_or_tensor(self):
        good = self.owned_binding()
        rng = torch.random.get_rng_state()
        binding, _ = load_all_layer_prompt_steering(**good)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(binding.completed_updates, 15)
        self.assertEqual(len(list(binding.probes.parameters())), 192)
        self.assertTrue(all(not p.requires_grad for p in binding.probes.parameters()))
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            load_all_layer_prompt_steering(**dict(good, probe_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_all_layer_prompt_steering(**dict(good, base_raw_bytes=good["base_raw_bytes"] + b"a"))
        with self.assertRaisesRegex(ValueError, "EMA disabled"):
            load_all_layer_prompt_steering(**self.owned_binding(ema=0.99))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_all_layer_prompt_steering(**self.owned_binding(invalid="63.probe.bias"))

    def test_zero_and_active_trace_are_native_and_ordered(self):
        binding, base = load_all_layer_prompt_steering(**self.owned_binding())
        owner = NativeOwner()
        session = Session(owner, base["sha256"])
        model = owner.native_model
        with owner.network.scaled(0):
            baseline = session.render(strength=0, max_new_tokens=3)
        original_rows = list(model.trace)
        model.trace.clear()
        with patch("arete_native_scheduling.all_layer.import_module", return_value=SimpleNamespace(NativeResponse=NativeOwner)):
            off = generate_all_layer(session, steering=binding, strength=0.0, max_new_tokens=3)
            off_rows = list(model.trace)
            model.trace.clear()
            on = generate_all_layer(session, steering=binding, strength=.5, max_new_tokens=3)
        self.assertEqual(off.native["new_token_ids"], baseline.native["new_token_ids"])
        self.assertTrue(all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                            for a, b in zip(original_rows, off_rows, strict=True)))
        self.assertTrue(torch.equal(original_rows[0][:, :-1], model.trace[0][:, :-1]))
        self.assertEqual(on.native["new_token_ids"], [8, 8, 8])
        meta = on.native["adaptive_steering"]
        self.assertEqual(meta["method"], "a_psr")
        self.assertEqual(meta["decoder_output_layers"], list(range(64)))
        self.assertEqual([step["cache_position"] for step in meta["answer_steps"]], [2, 3, 4])
        first = meta["answer_steps"][0]
        self.assertEqual([observed(x) for x in first["coefficient_float32_bits"][:2]], [.25, .125])
        self.assertEqual([observed(x) for x in first["effective_strength_float32_bits"][:2]], [.125, .0625])
        self.assertEqual(len(first["effective_strength_float32_bits"]), 64)
        self.assertEqual(owner.network.scale, .75)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.language_model.layers))

    def test_owner_raw_and_layer_placement_guard(self):
        good, packet = load_all_layer_prompt_steering(**self.owned_binding())
        wrong, _ = load_all_layer_prompt_steering(**self.owned_binding(devices=["cuda:0"] * 64))
        owner = NativeOwner()
        with patch("arete_native_scheduling.all_layer.import_module", return_value=SimpleNamespace(NativeResponse=NativeOwner)):
            with self.assertRaisesRegex(ValueError, "original RAW8 packet"):
                generate_all_layer(Session(owner, "f" * 64), steering=good, strength=.5, max_new_tokens=1)
            with self.assertRaisesRegex(ValueError, "layer placement"):
                generate_all_layer(Session(owner, packet["sha256"]), steering=wrong, strength=.5, max_new_tokens=1)
            with self.assertRaisesRegex(ValueError, "thinking disabled"):
                generate_all_layer(Session(owner, packet["sha256"]), steering=good, strength=.5,
                                   max_new_tokens=1, enable_thinking=True)
            model = owner.native_model
            model.omit_last = True
            with self.assertRaisesRegex(RuntimeError, "omitted an authored layer"):
                generate_all_layer(Session(owner, packet["sha256"]), steering=good, strength=.5, max_new_tokens=1)
            self.assertEqual(owner.network.scale, .75)
            self.assertFalse(model._forward_pre_hooks or model._forward_hooks)
            self.assertTrue(all(not layer._forward_hooks for layer in model.model.language_model.layers))

    @unittest.skipUnless(find_spec("arete_native95"), "Requires the retained archived native owner")
    def test_exact_archived_bridge_owns_existing_restore_and_render_context(self):
        from arete_native95.contracts import PROTOCOL
        from arete_native95.lifecycle import BACKEND, TEXT_CAPABILITIES
        from arete_native95.slider import AreteSlider
        from particle_sliders.inference import INFERENCE_V2, IDENTITY_V1
        from particle_sliders.pg09 import _hash

        owner = ArchivedNativeOwner()
        identity = dict(schema=IDENTITY_V1, model=owner.model_identity, protocol=PROTOCOL,
                        architectures={}, sources={}, network_schema={"raw": [[1], "torch.float32"]})
        base = dict(schema=INFERENCE_V2, selection="raw", network=owner.network.state_dict(),
                    origin=dict(cohort="synthetic", completed_steps=8, pg_version="0.11.0",
                                generator_arm="routed_particle_gan", training_manifest_sha256="a" * 64),
                    specification=dict(backend=BACKEND, configuration={"ema_decay": None},
                                       capabilities=asdict(TEXT_CAPABILITIES), response=True, identity=identity,
                                       factory_sources=dict(prepare=None, validate=None, preflight=None)))
        base["sha256"] = _hash(base)
        steering, decoded_base = load_all_layer_prompt_steering(**self.owned_binding(base=base))
        fixture = Session(owner, base["sha256"], eos_after=2)
        checks = []
        steps = []

        @contextmanager
        def aligned_text_device():
            def align(_model, args, kw):
                self.assertFalse(args)
                self.assertEqual(owner.network.scale, 0)
                self.assertEqual(len(owner.network._activation_stack), 2)
                steps.append((kw["past_key_values"], kw["input_ids"].shape, kw["attention_mask"].shape))
                for name in ("input_ids", "attention_mask", "position_ids"):
                    kw[name] = kw[name].to("cpu")
                return args, kw

            hook = owner.native_model.register_forward_pre_hook(align, prepend=True, with_kwargs=True)
            try:
                yield
            finally:
                hook.remove()

        with (patch("arete_native95.native.NativeResponse", ArchivedNativeOwner),
              patch("particle_sliders.inference.native_identity", return_value=identity),
              patch.object(Network, "scaled", AreteSlider.scaled),
              patch("arete_native95.finished_text.generate", side_effect=lambda native, **kw: fixture.render(**kw))):
            session = bind_archived_all_layer_session(owner, base_packet=decoded_base,
                                                      check=lambda: checks.append(True), render_context=aligned_text_device)
            result = generate_all_layer(session, steering=steering, strength=.5, max_new_tokens=3)
            self.assertEqual(result.native["new_token_ids"], [8, 2])
            self.assertEqual(len(result.native["adaptive_steering"]["answer_steps"]), 2)
            self.assertEqual([step[1:] for step in steps], [((1, 3), (1, 3)), ((1, 1), (1, 4))])
            self.assertIs(steps[0][0], steps[1][0])
            self.assertEqual(owner.network.scale, .75)
            self.assertFalse(owner.network._activation_stack or owner.native_model._forward_pre_hooks)
            self.assertGreaterEqual(len(checks), 3)
            owner.network.raw.add_(1)
            with self.assertRaisesRegex(RuntimeError, "changed before rendering"):
                generate_all_layer(session, steering=steering, strength=.5, max_new_tokens=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
