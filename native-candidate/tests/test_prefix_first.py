"""CPU composition of the archived writer, cached A-PSR and observer ingress."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

import test_all_layer as fixture
from arete_native_scheduling import prefix_first
from arete_native_scheduling import (
    bind_archived_all_layer_session, load_all_layer_prompt_steering, load_dose_matcher,
    load_steering_core, render_prefix_first,
)
from arete_native_scheduling.all_layer import _digest
from arete_native_scheduling.observer_ingress import _V3_OBSERVERS, prepare_blinded_trio


class PrefixFirst(unittest.TestCase):
    def setUp(self):
        if find_spec("arete_native95") is None:
            self.skipTest("Use the archived Arete PG0.11.0 package")
        shared, dose = (os.getenv(key) for key in ("ARETE_STEERING_CORE_WHEEL", "ARETE_DOSE_CORE_WHEEL"))
        if not shared or not dose:
            self.skipTest("Set both exact shared steering and dose wheel paths")
        from arete_native95.contracts import PROTOCOL
        from arete_native95.finished_text import TextOutput
        from arete_native95.lifecycle import BACKEND, TEXT_CAPABILITIES
        from particle_sliders.inference import IDENTITY_V1, INFERENCE_V2
        from particle_sliders.pg09 import _hash

        self.output_type = TextOutput
        self.core = load_steering_core(shared)
        self.matcher = load_dose_matcher(dose)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.owner = owner = fixture.ArchivedNativeOwner()
        owner.native_model.config.text_config.vocab_size = 16
        owner.native_model.config.text_config.max_position_embeddings = 32
        identity = dict(schema=IDENTITY_V1, model=owner.model_identity, protocol=PROTOCOL,
                        architectures={}, sources={}, network_schema={"raw": [[1], "torch.float32"]})
        base = dict(schema=INFERENCE_V2, selection="raw", network=owner.network.state_dict(),
                    origin=dict(cohort="synthetic", completed_steps=8, pg_version="0.11.0",
                                generator_arm="routed_particle_gan", training_manifest_sha256="a" * 64),
                    specification=dict(backend=BACKEND, configuration={"ema_decay": None},
                                       capabilities=asdict(TEXT_CAPABILITIES), response=True, identity=identity,
                                       factory_sources=dict(prepare=None, validate=None, preflight=None)))
        base["sha256"] = _hash(base)
        self.steering, decoded = load_all_layer_prompt_steering(**fixture.AllLayer.owned_binding(self, base=base))
        self.simulated_cache = fixture.Session(owner, base["sha256"])
        self.enterContext(patch("arete_native95.native.NativeResponse", fixture.ArchivedNativeOwner))
        self.enterContext(patch("particle_sliders.inference.native_identity", return_value=identity))
        self.render = self.enterContext(patch("arete_native95.finished_text.generate", side_effect=self.synthetic_text))
        self.session = bind_archived_all_layer_session(owner, base_packet=decoded,
                                                      check=lambda: None, render_context=nullcontext)
        self.ids = [1, 2, 3]
        self.arms = {}
        for index, arm in enumerate(("off", "adaptive", "constant")):
            stem = self.root / arm
            self.arms[arm] = dict(claim_path=str(stem) + ".claim", first_path=str(stem) + ".first",
                                  text_path=str(stem) + ".text", native_path=str(stem) + ".text.native.json",
                                  blind_id=f"tone-{index:012x}")
        binding = {}
        for name, (size, digest) in _V3_OBSERVERS.items():
            path = self.root / name
            path.write_bytes(b"f" * size)
            binding[name] = dict(path=str(path), bytes=size, sha256=digest)
        scope = dict(case_ref="synthetic-case", eligible_tokens=3, max_new_tokens=5, max_tokens=12, seed=126,
                     owner="arete_native95.native", native_model_sha256=_digest(owner.model_identity),
                     generation_config_sha256=_digest(owner.native_model.generation_config.to_dict()),
                     layer_devices=["cpu"] * 64, base_raw_sha256=self.steering.base_file_sha256,
                     base_packet_sha256=base["sha256"], authored_probe_sha256=self.steering.probe_sha256,
                     authored_plan_sha256=self.steering.plan_sha256, completed_authored_updates=15,
                     prompt_ids_sha256=_digest(self.ids), prompt_tokens=len(self.ids))
        self.registration = self.freeze(dict(schema="arete.a_psr.prefix_observer_registration.v1",
            scope=scope, arms=self.arms, ema_decay=None, dose_policy="nearest_binary32", adaptive_strength=.5,
            prompt="Describe the synthetic fixture.", integrity_output_contract="Complete text.", observer_binding=binding))

    def freeze(self, registration):
        data = json.dumps(registration).encode()
        path = self.root / "registration.json"
        path.write_bytes(data)
        return dict(path=str(path), bytes=len(data), sha256=sha256(data).hexdigest())

    def synthetic_text(self, owner, **request):
        self.assertIs(owner, self.owner)
        self.assertEqual(request["ids"], self.ids)
        self.assertEqual(request["seed"], 126)
        self.assertEqual(request["max_tokens"], 12)
        self.assertFalse(request["enable_thinking"])
        output = self.simulated_cache.render(**request)
        output.native.update(
            schema="arete-native-finished-text-v2", model=self.owner.model_identity,
            placement=self.owner.runtime.placement, strength=0.0, native_eos_token_id=[2],
            native_pad_token_id=0, native_bos_token_id=1,
            actual_generate_kwargs=dict(generation_config="fresh_per_call_explicit",
                                        model_inputs=["input_ids", "attention_mask"]),
            explicit_generation_config=dict(max_new_tokens=5, use_cache=True, do_sample=False, eos_token_id=[2]),
        )
        return self.output_type("Original synthetic text.\n", output.native)

    def first(self, arm, *, eos_after=None, ids=None):
        self.simulated_cache.eos_after = eos_after
        return render_prefix_first(self.registration, arm, self.session, steering=self.steering,
                                   ids=self.ids if ids is None else ids, matcher=self.matcher)

    def read(self, arm, part):
        return Path(self.arms[arm][f"{part}_path"]).read_text()

    def test_archived_v2_trio_roundtrips_original_firsts_and_natural_eos(self):
        refs = dict(off=self.first("off", eos_after=1), adaptive=self.first("adaptive", eos_after=4),
                    constant=self.first("constant", eos_after=3))
        result = prepare_blinded_trio(self.registration, refs, matcher=self.matcher)
        self.assertEqual(result["private"]["timing"]["status"], "ready_for_observation")
        self.assertEqual(result["private"]["missing_firsts"], [])
        for arm, length in (("off", 1), ("adaptive", 4), ("constant", 3)):
            intent = json.loads(self.read(arm, "claim"))
            self.assertEqual(intent["schema"], "arete.a_psr.prefix_native_intent.v1")
            self.assertEqual(intent["registration"], self.registration)
            self.assertEqual(intent["arm"], arm)
            self.assertEqual(self.read(arm, "text"), "Original synthetic text.\n")
            native = json.loads(self.read(arm, "native"))
            self.assertEqual(native["schema"], "arete-native-finished-text-v2")
            self.assertEqual(len(native["new_token_ids"]), length)
            self.assertEqual(native["new_token_ids"][-1], 2)
            self.assertEqual(self.first(arm), refs[arm])
        self.assertEqual(self.render.call_count, 3)
        self.assertEqual(self.owner.network.scale, .75)
        self.assertFalse(self.owner.native_model._forward_pre_hooks or self.owner.native_model._forward_hooks)

    def test_early_adaptive_eos_preserves_first_and_leaves_constant_unentered(self):
        off = self.first("off", eos_after=1)
        adaptive = self.first("adaptive", eos_after=2)
        self.assertIsNone(self.first("constant"))
        self.assertFalse(any(Path(self.arms["constant"][f"{part}_path"]).exists()
                             for part in ("claim", "text", "native", "first")))
        result = prepare_blinded_trio(self.registration, dict(off=off, adaptive=adaptive), matcher=self.matcher)
        self.assertEqual(result["private"]["timing"]["reason"], "adaptive_prefix_incomplete")
        self.assertEqual(len(result["integrity_inputs"]), 2)
        self.assertEqual(self.render.call_count, 2)

    def test_early_constant_eos_preserves_unmeasured_first(self):
        refs = dict(off=self.first("off", eos_after=1), adaptive=self.first("adaptive", eos_after=4),
                    constant=self.first("constant", eos_after=2))
        result = prepare_blinded_trio(self.registration, refs, matcher=self.matcher)
        self.assertEqual(result["private"]["timing"]["reason"], "constant_prefix_incomplete")
        self.assertEqual(len(result["integrity_inputs"]), 3)
        self.assertEqual(json.loads(self.read("constant", "native"))["new_token_ids"][-1], 2)

    def test_failed_native_call_keeps_durable_intent_and_rejects_replay(self):
        self.first("off")
        off_before = self.read("off", "first")

        def fail(*args, **kwargs):
            self.assertEqual(json.loads(self.read("adaptive", "claim"))["arm"], "adaptive")
            raise RuntimeError("simulated native failure")

        self.render.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "simulated native failure"):
            self.first("adaptive")
        claim = self.read("adaptive", "claim")
        with self.assertRaisesRegex(RuntimeError, "already entered"):
            self.first("adaptive")
        with self.assertRaisesRegex(RuntimeError, "already entered"):
            self.first("constant")
        self.assertEqual(self.read("adaptive", "claim"), claim)
        self.assertEqual(self.read("off", "first"), off_before)
        self.assertEqual(self.render.call_count, 2)
        self.assertFalse(self.owner.native_model._forward_pre_hooks or self.owner.native_model._forward_hooks)

    def test_early_validation_leaves_no_claim_on_request_or_path_change(self):
        with self.assertRaisesRegex(ValueError, "prompt IDs"):
            self.first("off", ids=[1, 2, 4])
        self.owner.native_model.generation_config.pad_token_id = 1
        with self.assertRaisesRegex(ValueError, "differs from registration"):
            self.first("off")
        self.owner.native_model.generation_config.pad_token_id = 0
        registered = json.loads(Path(self.registration["path"]).read_bytes())
        registered["arms"]["off"]["native_path"] = str(self.root / "fake-sidecar")
        self.registration = self.freeze(registered)
        with self.assertRaisesRegex(ValueError, "sidecar path"):
            self.first("off")
        self.assertFalse(Path(self.arms["off"]["claim_path"]).exists())
        self.assertEqual(self.render.call_count, 0)

    def test_partial_original_write_preserves_text_and_blocks_replay(self):
        def partial(output, path):
            path.write_text(output.text)
            raise OSError("simulated sidecar failure")

        with patch.object(self.output_type, "save", partial):
            with self.assertRaisesRegex(OSError, "simulated sidecar failure"):
                self.first("off")
        self.assertEqual(self.read("off", "text"), "Original synthetic text.\n")
        with self.assertRaisesRegex(RuntimeError, "already entered"):
            self.first("off")
        self.assertFalse(Path(self.arms["off"]["first_path"]).exists())
        self.assertEqual(self.render.call_count, 1)

    def test_unusable_native_receipt_preserves_original_output_without_first(self):
        def unexpected_terminal(*args, **kwargs):
            output = self.synthetic_text(*args, **kwargs)
            output.native["native_eos_token_id"] = [1]
            return output

        self.render.side_effect = unexpected_terminal
        with self.assertRaisesRegex(ValueError, "registered RAW arm or answer clock"):
            self.first("off", eos_after=1)
        self.assertEqual(self.read("off", "text"), "Original synthetic text.\n")
        self.assertEqual(json.loads(self.read("off", "native"))["new_token_ids"], [2])
        self.assertFalse(Path(self.arms["off"]["first_path"]).exists())
        with self.assertRaisesRegex(RuntimeError, "already entered"):
            self.first("off")
        self.assertEqual(self.render.call_count, 1)

    def test_competing_same_arm_claims_dispatch_once(self):
        barrier = Barrier(2)
        write_receipt = prefix_first._write_receipt

        def race_at_claim(path, record):
            if path == Path(self.arms["off"]["claim_path"]):
                barrier.wait(timeout=5)
            return write_receipt(path, record)

        with patch.object(prefix_first, "_write_receipt", race_at_claim), ThreadPoolExecutor(2) as pool:
            contenders = [pool.submit(self.first, "off") for _ in range(2)]
        result = [contender.result() for contender in contenders if contender.exception() is None]
        failed = [contender.exception() for contender in contenders if contender.exception() is not None]
        self.assertEqual(len(result), 1)
        self.assertEqual(len(failed), 1)
        self.assertIsInstance(failed[0], FileExistsError)
        self.assertEqual(self.first("off"), result[0])
        self.assertEqual(self.render.call_count, 1)
