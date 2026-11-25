"""Synthetic registered firsts for the settled Arete observer handoff."""

from contextlib import contextmanager
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from struct import pack
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arete_native_scheduling import load_dose_matcher, match_all_layer_constant
from arete_native_scheduling.all_layer import _constant_dose_receipt
from arete_native_scheduling.observer_ingress import prepare_blinded_trio
from arete_native_scheduling.steering_core import CORE_REVISION


class ObserverIngress(unittest.TestCase):
    def setUp(self):
        wheel = os.getenv("ARETE_DOSE_CORE_WHEEL")
        if not wheel:
            self.skipTest("Set ARETE_DOSE_CORE_WHEEL to the wheel with the pinned dose source")
        self.matcher = load_dose_matcher(wheel)
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.traces = {}

    def freeze(self, path, content):
        data = content.encode() if isinstance(content, str) else json.dumps(content).encode()
        with (self.root / path).open("xb") as file:
            file.write(data)
        return dict(path=str(self.root / path), bytes=len(data), sha256=sha256(data).hexdigest())

    def register(self, dose_policy="exact", *, qualified=True):
        # The handoff compares the prior adopted metadata. The original guard
        # checks the qualification bodies later in its own read aperture.
        binding = {}
        for name, size, digest in (
            ("calibration_plan", 14613, "148fe16a1e55487e5359121ca58606c269cab77021227798810893ad71d6bea6"),
            ("tone_qualification", 7415, "be4162616893fed5490e9b67d1c65f12f854d97d450e99f7ef98996d4fb8d92a"),
            ("guarded_calibration", 225, "c153252738897033d3a748585d9c12d3a69a8d8534ec524e3f69e41b7048d6ca"),
        ):
            placeholder = self.freeze(name, "x" * size)
            binding[name] = dict(placeholder, sha256=digest)
        if not qualified:
            binding["tone_qualification"]["sha256"] = "f" * 64
        self.artifacts = dict(base_raw_sha256="a" * 64, base_packet_sha256="b" * 64,
                              authored_probe_sha256="c" * 64, authored_plan_sha256="d" * 64)
        self.scope = dict(case_ref="registered-case", eligible_tokens=3, max_new_tokens=4, max_tokens=512,
                          seed=126, prompt_tokens=4, prompt_ids_sha256="e" * 64, **self.artifacts)
        self.arms = {}
        for arm, opaque in (("off", "9"), ("adaptive", "1"), ("constant", "5")):
            self.arms[arm] = dict(blind_id="tone-" + opaque * 12)
            self.arms[arm].update({f"{kind}_path": str(self.root / f"{arm}.{kind}")
                                   for kind in ("claim", "first", "native", "text")})
        self.registration = self.freeze("registration", dict(schema="arete.a_psr.prefix_observer_registration.v1",
            scope=self.scope, ema_decay=None, dose_policy=dose_policy, adaptive_strength=.5,
            prompt="Describe the card.", integrity_output_contract="Original complete native text.",
            observer_binding=binding, arms=self.arms))

    def first(self, arm, gains, *, mutation=None, claim_mutation=None, text="The card reads “Wonderful!”.\n"):
        intent = dict(schema="arete.a_psr.prefix_native_intent.v1", registration=self.registration,
                      arm=arm, case_ref=self.scope["case_ref"], ema_decay=None)
        if claim_mutation:
            claim_mutation(intent)
        claim = self.freeze(f"{arm}.claim", intent)
        output = self.freeze(f"{arm}.text", text)
        if arm == "constant":
            plan = match_all_layer_constant(self.traces["adaptive"], matcher=self.matcher)
            rows = [plan.strengths if i < 3 else (0.0,) * 64 for i in range(len(gains))]
        else:
            rows = [(gain,) + (0.0,) * 63 for gain in gains]
        answer_steps = [dict(output_index=i, cache_position=3+i,
                             effective_strength_float32_bits=[pack(">f", gain).hex() for gain in row])
                        for i, row in enumerate(rows)]
        trace = dict(core_revision=CORE_REVISION, method="a_psr_constant" if arm == "constant" else "a_psr",
                     nominal_strength={"off": 0.0, "adaptive": .5, "constant": None}[arm], raw_only=True,
                     ema_decay=None, position_scope="answer_only", intervention="residual", prefix_tokens=4,
                     prefix_dose_scope=self.scope, eligible_prefix_complete=len(rows) >= 3,
                     decoder_output_layers=list(range(64)), answer_steps=answer_steps, **self.artifacts)
        if arm == "constant":
            trace["dose_comparison"] = _constant_dose_receipt(trace, plan)
        native = dict(schema="arete095-native-finished-text-v1", selection="raw", weight_EMA=None,
                      enable_thinking=False, strength=0.0, seed=126, prompt_tokens=4, native_eos_token_id=9,
                      new_token_ids=[1] * (len(rows) - 1) + [9], actual_generate_kwargs=dict(max_new_tokens=4),
                      adaptive_steering=trace, model=dict(identity="fixture"), placement="fixture")
        if mutation:
            mutation(native)
        receipt = self.freeze(f"{arm}.native", native)
        self.traces[arm] = trace
        return self.freeze(f"{arm}.first", dict(claim=claim, text=output, native=receipt))

    def prepare(self, first_refs):
        return prepare_blinded_trio(self.registration, first_refs, matcher=self.matcher)

    def trio(self, gains=(.5,.5,.5,0)):
        return dict(off=self.first("off", [0]), adaptive=self.first("adaptive", gains),
                    constant=self.first("constant", [0,0,0]))

    def test_exact_trio_blinds_arms_and_preserves_complete_original_text(self):
        self.register()
        firsts = self.trio()
        result = self.prepare(firsts)
        self.assertEqual(result["private"]["timing"]["status"], "ready_for_observation")
        self.assertTrue(result["private"]["timing"]["dose"]["exact_dose_equal"])
        self.assertEqual(result["private"]["missing_firsts"], [])
        rows = result["tone_bank"]["rows"]
        self.assertEqual([row["id"] for row in rows], [self.arms[arm]["blind_id"] for arm in ("adaptive", "constant", "off")])
        for tone, integrity in zip(rows, result["integrity_inputs"], strict=True):
            self.assertEqual(set(tone), {"id", "prompt", "output"})
            self.assertEqual(set(integrity), {"file", "media_type", "output_contract"})
            self.assertEqual(tone["output"], integrity["file"])
            self.assertEqual(Path(integrity["file"]["path"]).read_text(), "The card reads “Wonderful!”.\n")

    def test_early_adaptive_eos_and_missing_constant_keep_original_firsts(self):
        self.register()
        result = self.prepare(dict(off=self.first("off", [0]), adaptive=self.first("adaptive", [.5])))
        self.assertEqual(result["private"]["timing"]["reason"], "adaptive_prefix_incomplete")
        self.assertEqual(result["private"]["missing_firsts"], ["constant"])
        self.assertEqual(len(result["integrity_inputs"]), 2)

    def test_archived_pg0110_v2_phase_receipt_uses_explicit_generation_policy(self):
        self.register()

        def phase_v2(native):
            native.update(
                schema="arete-native-finished-text-v2", native_eos_token_id=[9, 10],
                native_bos_token_id=2, native_pad_token_id=9, pretrained_quality_qualified=False,
                actual_generate_kwargs=dict(generation_config="fresh_per_call_explicit",
                                            model_inputs=["input_ids", "attention_mask"]),
                explicit_generation_config=dict(do_sample=False, num_beams=1, num_return_sequences=1,
                    use_cache=True, max_new_tokens=4, min_new_tokens=0, min_length=0, temperature=1.,
                    top_k=50, top_p=1., repetition_penalty=1., no_repeat_ngram_size=0,
                    return_dict_in_generate=True, output_scores=False, output_logits=False,
                    output_attentions=False, output_hidden_states=False, eos_token_id=[9, 10],
                    bos_token_id=2, pad_token_id=9),
            )

        refs = dict(off=self.first("off", [0], mutation=phase_v2),
                    adaptive=self.first("adaptive", [.5,.5,.5,0], mutation=phase_v2),
                    constant=self.first("constant", [0,0,0], mutation=phase_v2))
        result = self.prepare(refs)
        self.assertEqual(result["private"]["timing"]["status"], "ready_for_observation")
        self.assertEqual(len(result["integrity_inputs"]), 3)

    def test_early_constant_eos_is_unmeasured_even_when_gains_are_correct(self):
        self.register()
        refs = dict(off=self.first("off", [0]), adaptive=self.first("adaptive", [.5,.5,.5]),
                    constant=self.first("constant", [0,0]))
        timing = self.prepare(refs)["private"]["timing"]
        self.assertEqual(timing["reason"], "constant_prefix_incomplete")
        self.assertIsNone(timing["dose"]["exact_dose_equal"])
        self.assertEqual(timing["dose"]["layers"][0]["signed_residual"], "-1/2")

    def test_rounded_dose_remains_unmeasured_under_exact_policy(self):
        self.register()
        timing = self.prepare(self.trio((.5,.5,0,0)))["private"]["timing"]
        self.assertEqual(timing["reason"], "nearest_binary32_residual_under_exact_only_policy")
        self.assertNotEqual(timing["dose"]["layers"][0]["signed_residual"], "0")

    def test_registered_nearest_policy_uses_the_verified_native_rounding(self):
        self.register("nearest_binary32")
        timing = self.prepare(self.trio((.5,.5,0,0)))["private"]["timing"]
        self.assertEqual(timing["status"], "ready_for_observation")
        self.assertFalse(timing["dose"]["exact_dose_equal"])

    def test_changed_gain_cannot_pass_with_the_same_integrated_dose(self):
        self.register()
        off, adaptive = self.first("off", [0]), self.first("adaptive", [.5,.5,.5])

        def changed_gain(native):
            rows = native["adaptive_steering"]["answer_steps"]
            rows[0]["effective_strength_float32_bits"][0] = pack(">f", .25).hex()
            rows[1]["effective_strength_float32_bits"][0] = pack(">f", .75).hex()

        constant = self.first("constant", [0,0,0], mutation=changed_gain)
        with self.assertRaisesRegex(ValueError, "gain other than"):
            self.prepare(dict(off=off, adaptive=adaptive, constant=constant))

    def test_wrong_receipt_scope_or_old_claim_refuses_ingress(self):
        self.register()
        broken = self.first("off", [0], claim_mutation=lambda claim: claim.update(arm="adaptive"))
        with self.assertRaisesRegex(ValueError, "exact registration and arm"):
            self.prepare(dict(off=broken))
        adaptive = self.first("adaptive", [.5,.5,.5], mutation=lambda native: native.update(seed=1))
        with self.assertRaisesRegex(ValueError, "registered RAW arm or answer clock"):
            self.prepare(dict(adaptive=adaptive))
        with self.assertRaisesRegex(ValueError, "First path differs"):
            self.prepare(dict(off=adaptive))

    def test_unqualified_observer_fails_closed(self):
        self.register(qualified=False)
        first = self.first("off", [0])
        with self.assertRaisesRegex(ValueError, "guarded v3 tone and separate"):
            self.prepare(dict(off=first))

    def test_missing_claim_cannot_lend_a_finished_first(self):
        self.register()
        first = self.first("off", [0])
        (self.root / "off.claim").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare(dict(off=first))

    def test_text_replaced_at_open_with_older_mtime_is_rejected_before_read(self):
        self.register()
        first = self.first("off", [0])
        text = str(self.root / "off.text")
        before_claim = (self.root / "off.claim").stat().st_mtime_ns - 1
        real_open, real_fdopen = os.open, os.fdopen

        def redate_on_open(path, flags):
            if str(path) == text:
                os.utime(path, ns=(before_claim, before_claim))
            return real_open(path, flags)

        class Unreadable:
            def __init__(self, stream):
                self.fileno = stream.fileno

            def read(self, _size):
                raise AssertionError("Stale text body was read")

        @contextmanager
        def stream(fd, mode):
            target = os.readlink(f"/proc/self/fd/{fd}")
            with real_fdopen(fd, mode) as file:
                yield Unreadable(file) if target == text else file

        with (patch("arete_native_scheduling.observer_ingress.os.open", side_effect=redate_on_open),
              patch("arete_native_scheduling.observer_ingress.os.fdopen", side_effect=stream),
              self.assertRaisesRegex(ValueError, "chronology")):
            self.prepare(dict(off=first))

    def test_existing_v3_masks_only_its_view_of_the_same_input(self):
        script = os.getenv("ARETE_V3_TONE_SOURCE")
        if not script:
            self.skipTest("Set ARETE_V3_TONE_SOURCE to the existing instrument's rate_tone.py")
        self.register()
        row = self.prepare(dict(off=self.first("off", [0])))["tone_bank"]["rows"][0]
        spec = spec_from_file_location("exact_v3_tone_fixture", script)
        module = module_from_spec(spec)

        class Processor:
            def apply_chat_template(self, messages, **_opts):
                return messages[-1]["content"][0]["text"]

            def __call__(self, *, text, **_opts):
                return dict(input_ids=text)

        with patch.object(sys, "path", [str(Path(script).parent), *sys.path]):
            spec.loader.exec_module(module)
            _, result = module.input_for(SimpleNamespace(processor=Processor()), dict(instruction="Synthetic fixture."), row)
        view = json.loads(result["input_ids"][0])
        self.assertEqual(view["finished_text"], "The card reads [reported wording].\n")
        self.assertEqual(Path(row["output"]["path"]).read_text(), "The card reads “Wonderful!”.\n")


if __name__ == "__main__":
    unittest.main()
