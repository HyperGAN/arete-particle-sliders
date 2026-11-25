"""Synthetic returned v3 observer receipts and a full development grid."""

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import test_observer_ingress as fixture
from arete_native_scheduling import load_dose_matcher
from arete_native_scheduling.development_evidence import (
    freeze_prefix_development, read_prefix_development_case, reduce_prefix_development,
)
from arete_native_scheduling.observer_ingress import _V3_OBSERVERS


class DevelopmentEvidence(unittest.TestCase):
    def setUp(self):
        wheel = os.getenv("ARETE_DOSE_CORE_WHEEL")
        if not wheel:
            self.skipTest("Set ARETE_DOSE_CORE_WHEEL to the pinned dose source")
        self.matcher = load_dose_matcher(wheel)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = {"profile": {"sha256": "d" * 64}}
        self.integrity_ref = self.freeze(self.root / "qualification", {
            "qualified": True, "scope": "complete text only", "profile": self.profile,
        })
        self.bindings = dict(
            calibration_plan=self.freeze(self.root / "calibration", {
                "original_BF16_integrity_qualification": self.integrity_ref,
            }),
            tone_qualification=self.freeze(self.root / "tone-qualification", {"qualified": True}),
            guarded_calibration=self.freeze(self.root / "guarded-calibration", {"returncode": 0, "reaped": True}),
        )
        expectations = {key: (value["bytes"], value["sha256"]) for key, value in self.bindings.items()}
        self.enterContext(patch.dict(_V3_OBSERVERS, expectations, clear=True))
        self.cells = {}
        self.frozen_path = self.root / "development-first.json"

    def freeze(self, path, body):
        data = json.dumps(body).encode()
        with path.open("xb") as stream:
            stream.write(data)
        return dict(path=str(path), bytes=len(data), sha256=sha256(data).hexdigest())

    def panel(self, *, alphas=(.25, .5), cases=("case-a", "case-b"), role="development"):
        trials = []
        for alpha in alphas:
            registrations = {}
            for case in cases:
                unit = fixture.ObserverIngress()
                unit.matcher = self.matcher
                unit.root = self.root / f"{case}-{alpha}"
                unit.root.mkdir()
                unit.traces = {}
                unit.register()
                registration = json.loads(Path(unit.registration["path"]).read_bytes())
                unit.scope.update(case_ref=case, completed_authored_updates=30, owner="arete_native95.native",
                                  native_model_sha256="a" * 64, generation_config_sha256="b" * 64,
                                  layer_devices=["cpu"] * 64)
                registration.update(scope=unit.scope, adaptive_strength=alpha, observer_binding=self.bindings)
                unit.registration = self.freeze(unit.root / "registered", registration)
                self.cells[alpha, case] = unit
                registrations[case] = unit.registration
            trials.append(dict(alpha=alpha, registrations=registrations))
        return self.freeze(self.root / "panel.json", dict(schema="arete.a_psr.prefix_development_registration.v1",
            case_ids=list(cases), trials=trials, case_role=role, ema_decay=None))

    def cell(self, alpha, case, tones, *, integrity=None, early=None, mutate=None):
        unit = self.cells[alpha, case]
        verdicts = integrity or {}
        firsts = dict(off=unit.first("off", [0], text="Synthetic original baseline.\n"))

        def exact_nominal(native):
            native["adaptive_steering"]["nominal_strength"] = alpha

        firsts["adaptive"] = unit.first("adaptive", [.5] if early == "adaptive" else [.5,.5,.5],
                                        mutation=exact_nominal, text="Synthetic adaptive output.\n")
        if early != "adaptive":
            firsts["constant"] = unit.first("constant", [0,0] if early == "constant" else [0,0,0],
                                            text="Synthetic matched constant output.\n")
        root = unit.root / "observer"
        (root / "tone-firsts").mkdir(parents=True)
        (root / "firsts").mkdir()
        ratings = {}
        for arm, grade in tones.items():
            blind = unit.arms[arm]["blind_id"]
            source = json.loads(Path(firsts[arm]["path"]).read_bytes())["text"]
            status = verdicts.get(arm, "pass")
            call = dict(blind_id=blind, full_text=source)
            categorical = dict(output_source=source, media_type="text", native_contract="Original complete native text.",
                               actual_instrument_source=self.profile,
                               verdict=dict(status=status, categorical_failure={"pass": False, "fail": True,
                                                                              "unmeasured": None}[status]))
            if mutate is not None:
                mutate(arm, call, categorical)
            call_ref = self.freeze(root / "tone-firsts" / f"{blind}-committed.json", call)
            tone = self.freeze(root / "tone-firsts" / f"{blind}-first.json", dict(call=call_ref, error=None,
                result=None if grade is None else dict(blind_id=blind, intensity=grade, visible_evidence="Synthetic.")))
            integrity_first = self.freeze(root / "firsts" / f"{blind}-first.json", categorical)
            ratings[blind] = dict(tone=grade, tone_first=tone, integrity=status, integrity_first=integrity_first)
        bank = self.freeze(root / "blinded-firsts.json", dict(ratings=ratings,
            tone_calibration=self.bindings["tone_qualification"], integrity_qualification=self.integrity_ref))
        guard = self.freeze(root / "guard-terminal.json", dict(returncode=0, reaped=True))
        return dict(native_firsts=firsts, blinded_firsts=bank, guard_terminal=guard)

    def test_grid_uses_equal_case_adaptive_gain_and_freezes_once(self):
        panel = self.panel()
        scores = {
            .25: {"case-a": self.cell(.25, "case-a", dict(off=1, adaptive=4, constant=3)),
                  "case-b": self.cell(.25, "case-b", dict(off=1, adaptive=1, constant=3))},
            .5: {"case-a": self.cell(.5, "case-a", dict(off=1, adaptive=2, constant=1)),
                 "case-b": self.cell(.5, "case-b", dict(off=1, adaptive=2, constant=1))},
        }
        summary = freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertEqual(summary["selected_alpha"], .25)
        self.assertEqual(summary["trials"][0]["mean_gain"], 1.5)
        self.assertEqual(summary["trials"][0]["matched_constant"]["mean_gain"], 2)
        self.assertEqual(summary["trials"][1]["mean_gain"], 1)
        frozen = self.frozen_path.read_bytes()
        self.assertEqual(summary["frozen"]["sha256"], sha256(frozen).hexdigest())
        with self.assertRaises(FileExistsError):
            freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertEqual(frozen, self.frozen_path.read_bytes())

    def test_missing_case_has_no_partial_mean(self):
        panel = self.panel(alphas=(.5,))
        scores = {.5: {"case-a": self.cell(.5, "case-a", dict(off=1, adaptive=4, constant=3))}}
        r = freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertIsNone(r["trials"][0]["mean_gain"])
        self.assertIsNone(r["frozen"])
        self.assertFalse(self.frozen_path.exists())

    def test_complete_intensity_survives_unmeasured_timing(self):
        panel = self.panel(alphas=(.5,), cases=("case-a",))
        cell = self.cell(.5, "case-a", dict(off=1, adaptive=4, constant=3), early="constant")
        r = freeze_prefix_development(panel, {.5: {"case-a": cell}}, self.frozen_path, matcher=self.matcher)
        trial = r["trials"][0]
        self.assertEqual(trial["mean_gain"], 3)
        self.assertEqual(trial["status"], "unmeasured")
        self.assertEqual(trial["matched_constant"]["mean_gain"], 2)
        self.assertEqual(trial["cases"][0]["timing"]["reason"], "constant_prefix_incomplete")
        self.assertIsNone(r["frozen"])

    def test_positive_tie_has_no_tiebreak(self):
        panel = self.panel(cases=("case-a",))
        scores = {alpha: {"case-a": self.cell(alpha, "case-a", dict(off=1, adaptive=2, constant=4))}
                  for alpha in (.25, .5)}
        r = freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertEqual(r["best_alphas"], [.25, .5])
        self.assertEqual(r["freeze_reason"], "tied_development_alphas")
        self.assertFalse(self.frozen_path.exists())

    def test_constant_without_a_score_stays_separate_when_native_timing_matches(self):
        panel = self.panel(alphas=(.5,), cases=("case-a",))
        cell = self.cell(.5, "case-a", dict(off=1, adaptive=2))
        r = freeze_prefix_development(panel, {.5: {"case-a": cell}}, self.frozen_path, matcher=self.matcher)
        self.assertEqual(r["selected_alpha"], .5)
        self.assertEqual(r["trials"][0]["matched_constant"], dict(status="unmeasured", mean_gain=None))

    def test_nonpositive_tone_is_measured_but_cannot_freeze(self):
        panel = self.panel(cases=("case-a",))
        scores = {
            .25: {"case-a": self.cell(.25, "case-a", dict(off=2, adaptive=1, constant=4))},
            .5: {"case-a": self.cell(.5, "case-a", dict(off=2, adaptive=2, constant=4))},
        }
        r = freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertEqual([trial["status"] for trial in r["trials"]], ["pass", "pass"])
        self.assertEqual(r["best_mean_gain"], 0)
        self.assertEqual(r["freeze_reason"], "no_positive_development_gain")
        self.assertFalse(self.frozen_path.exists())

    def test_adaptive_catastrophe_keeps_intensity_and_cannot_win(self):
        panel = self.panel(cases=("case-a",))
        scores = {
            .25: {"case-a": self.cell(.25, "case-a", dict(off=1, adaptive=4, constant=3),
                                       integrity=dict(adaptive="fail"))},
            .5: {"case-a": self.cell(.5, "case-a", dict(off=1, adaptive=2, constant=4),
                                     integrity=dict(constant="fail"))},
        }
        r = freeze_prefix_development(panel, scores, self.frozen_path, matcher=self.matcher)
        self.assertEqual(r["selected_alpha"], .5)
        self.assertEqual(r["trials"][0]["mean_gain"], 3)
        self.assertEqual(r["trials"][0]["status"], "fail")
        self.assertEqual(r["trials"][1]["matched_constant"]["status"], "fail")

    def test_blind_first_must_bind_the_registered_text(self):
        panel = self.panel(alphas=(.5,), cases=("case-a",))

        def wrong_tone(arm, call, categorical):
            if arm == "adaptive":
                call["full_text"]["path"] += ".other"

        cell = self.cell(.5, "case-a", dict(off=1, adaptive=4), mutate=wrong_tone)
        with self.assertRaisesRegex(ValueError, "exact registered full text"):
            reduce_prefix_development(panel, {.5: {"case-a": cell}}, matcher=self.matcher)

    def test_integrity_must_preserve_the_original_contract(self):
        self.panel(alphas=(.5,), cases=("case-a",))

        def wrong_integrity(arm, call, categorical):
            categorical["native_contract"] = "Wrong contract."

        cell = self.cell(.5, "case-a", dict(off=1, adaptive=4), mutate=wrong_integrity)
        unit = self.cells[.5, "case-a"]
        with self.assertRaisesRegex(ValueError, "qualified complete text"):
            read_prefix_development_case(unit.registration, cell["native_firsts"], matcher=self.matcher,
                                         bank_ref=cell["blinded_firsts"], guard_ref=cell["guard_terminal"])

    def test_heldout_role_cannot_enter_development(self):
        panel = self.panel(alphas=(.5,), cases=("case-a",), role="heldout")
        with self.assertRaisesRegex(ValueError, "development-only"):
            reduce_prefix_development(panel, {}, matcher=self.matcher)

    def test_guard_failure_cannot_admit_scored_firsts(self):
        panel = self.panel(alphas=(.5,), cases=("case-a",))
        cell = self.cell(.5, "case-a", dict(off=1, adaptive=4))
        path = Path(cell["guard_terminal"]["path"]).with_name("failed-guard.json")
        cell["guard_terminal"] = self.freeze(path, dict(returncode=1, reaped=True))
        with self.assertRaisesRegex(ValueError, "guard did not settle cleanly"):
            reduce_prefix_development(panel, {.5: {"case-a": cell}}, matcher=self.matcher)
