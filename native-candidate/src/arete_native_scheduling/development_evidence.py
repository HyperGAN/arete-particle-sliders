"""Join settled Arete observer firsts and freeze a measured development alpha."""

from math import isfinite
from pathlib import Path

from .observer_ingress import _ARMS, _pinned_stat, _read_registration, _verified_json, prepare_blinded_trio
from .prefix_first import _write_receipt


_PANEL = "arete.a_psr.prefix_development_registration.v1"
_SETTLED_TIMING_REASONS = {
    "adaptive_prefix_incomplete",
    "constant_prefix_incomplete",
    "nearest_binary32_residual_under_exact_only_policy",
    "observer_input_outside_existing_limits",
}


def _observer_scope(prepared, bank_ref, guard_ref):
    if Path(bank_ref["path"]).parent != Path(guard_ref["path"]).parent:
        raise ValueError("Observer firsts and terminal must belong to the same guarded scope")
    bank, bank_time = _verified_json(bank_ref)
    guard, _ = _verified_json(guard_ref, not_before=bank_time)
    if type(guard.get("returncode")) is not int or guard["returncode"] != 0 or guard.get("reaped") is not True:
        raise ValueError("The observer guard did not settle cleanly")
    binding = prepared["observer_binding"]
    plan, _ = _verified_json(binding["calibration_plan"])
    integrity_ref = plan["original_BF16_integrity_qualification"]
    qualification, _ = _verified_json(integrity_ref)
    if (bank["tone_calibration"] != binding["tone_qualification"]
            or bank["integrity_qualification"] != integrity_ref
            or qualification.get("qualified") is not True or qualification.get("scope") != "complete text only"):
        raise ValueError("Observer firsts need the registered v3 and qualified complete-text integrity")
    return bank["ratings"], qualification["profile"], bank_time


def _observer_first(blind, arm, integrity_input, rating, profile, bank_time, root):
    for key, folder in (("tone_first", "tone-firsts"), ("integrity_first", "firsts")):
        if Path(rating[key]["path"]).parent != root / folder:
            raise ValueError("Observer first is outside its returned guarded scope")
    _, native_first = _pinned_stat(arm["first"])
    chronology = dict(not_before=native_first.st_mtime_ns, not_after=bank_time)
    tone, tone_time = _verified_json(rating["tone_first"], **chronology)
    if (Path(tone["call"]["path"]).parent != root / "tone-firsts"
            or Path(tone["call"]["path"]).name != blind + "-committed.json"
            or Path(rating["tone_first"]["path"]).name != blind + "-first.json"):
        raise ValueError("Tone call differs from its registered blind ID")
    call, _ = _verified_json(tone["call"], not_before=native_first.st_mtime_ns, not_after=tone_time)
    if call.get("blind_id") != blind or call.get("full_text") != arm["text"]:
        raise ValueError("Tone first did not observe the exact registered full text")
    result = tone["result"]
    intensity = None
    if result is not None:
        intensity = result.get("intensity")
        if (tone.get("error") is not None or result.get("blind_id") != blind
                or type(intensity) is not int or not 0 <= intensity <= 4):
            raise ValueError("Invalid v3 tone first")
    if intensity != rating.get("tone") or type(intensity) is not type(rating.get("tone")):
        raise ValueError("Frozen intensity differs from its v3 tone first")

    integrity, _ = _verified_json(rating["integrity_first"], **chronology)
    if integrity_input is None:
        raise ValueError("This text is outside the qualified integrity input limits")
    # The original readout caches integrity by identical output bytes in its
    # current guarded scope. The complete text contract and profile also bind it.
    source = integrity["output_source"]
    expected = integrity_input["file"]
    status = integrity["verdict"]["status"]
    categorical = {"pass": False, "fail": True, "unmeasured": None}
    if (any(source.get(key) != expected[key] for key in ("bytes", "sha256"))
            or integrity.get("media_type") != integrity_input["media_type"]
            or integrity.get("native_contract") != integrity_input["output_contract"]
            or integrity.get("actual_instrument_source") != profile
            or status not in categorical or integrity["verdict"].get("categorical_failure") is not categorical[status]
            or status != rating.get("integrity")):
        raise ValueError("Integrity first did not observe the corresponding qualified complete text")
    return dict(tone=intensity, integrity=status, tone_first=rating["tone_first"],
                integrity_first=rating["integrity_first"], output=arm["text"])


def read_prefix_development_case(registration_ref, native_firsts, *, matcher, bank_ref=None, guard_ref=None):
    """Read scalar and categorical evidence from the original guarded observers."""
    prepared = prepare_blinded_trio(registration_ref, native_firsts, matcher=matcher)
    timing = prepared["private"]["timing"]
    arms = prepared["private"]["arms"]
    observations = {arm: dict(tone=None, integrity="unmeasured") for arm in _ARMS}
    if (bank_ref is None) != (guard_ref is None):
        raise ValueError("Provide both the frozen blinded firsts and their observer guard terminal")
    if bank_ref is not None:
        ratings, profile, bank_time = _observer_scope(prepared, bank_ref, guard_ref)
        for arm, record in arms.items():
            blind = record["blind_id"]
            if blind not in ratings:
                continue
            if record["tone_input_index"] is None:
                raise ValueError("This text is outside the qualified tone input limits")
            index = record["integrity_input_index"]
            integrity = None if index is None else prepared["integrity_inputs"][index]
            observations[arm] = _observer_first(blind, record, integrity, ratings[blind], profile, bank_time,
                                                Path(bank_ref["path"]).parent)
    return dict(case_ref=prepared["private"]["case_ref"], observations=observations, timing=timing,
                registration=registration_ref, native_firsts=native_firsts, blinded_firsts=bank_ref,
                guard_terminal=guard_ref)


def _cohort(panel):
    if panel.get("schema") != _PANEL or panel.get("case_role") != "development" or panel["ema_decay"] is not None:
        raise ValueError("Expected an independently registered development-only RAW panel")
    cases = panel["case_ids"]
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, str) or not case.strip() for case in cases) or len(set(cases)) != len(cases)):
        raise ValueError("Expected one nonempty roster of distinct development cases")
    trials = {}
    paths = set()
    for trial in panel["trials"]:
        alpha = trial["alpha"]
        regs = trial["registrations"]
        if (type(alpha) not in (int, float) or not isfinite(alpha) or alpha in trials
                or set(regs) != set(cases)):
            raise ValueError("Register distinct finite alphas on the same complete development roster")
        trials[alpha] = regs
        for ref in regs.values():
            if ref["path"] in paths:
                raise ValueError("Every case and alpha needs its own exact registration")
            paths.add(ref["path"])
    if not trials:
        raise ValueError("Expected a preregistered development alpha grid")
    return cases, trials


def _case_identity(registration):
    return tuple(registration[key] for key in ("scope", "prompt", "observer_binding",
                                               "integrity_output_contract", "dose_policy"))


def _case_status(evidence, arm):
    off, on = (evidence["observations"][key] for key in ("off", arm))
    if on["integrity"] == "fail":
        return "fail", f"{arm}_catastrophic_integrity_failure"
    if evidence["timing"]["status"] != "ready_for_observation":
        return "unmeasured", evidence["timing"]["reason"]
    if off["integrity"] != "pass" or on["integrity"] != "pass":
        return "unmeasured", f"off_or_{arm}_integrity_not_pass"
    if off["tone"] is None or on["tone"] is None:
        return "unmeasured", f"missing_off_or_{arm}_intensity"
    return "pass", None


def _trial_status(statuses):
    return "fail" if "fail" in statuses else "unmeasured" if "unmeasured" in statuses else "pass"


def reduce_prefix_development(panel_ref, evidence, *, matcher):
    """Reduce settled evidence on the exact declared development grid."""
    from particle_sliders.alpha_search import matched_effect_mean

    panel, panel_time = _verified_json(panel_ref)
    cases, trials = _cohort(panel)
    if not isinstance(evidence, dict) or set(evidence) - trials.keys():
        raise ValueError("Only registered development alphas may enter this reduction")
    results, identities, off_sources, off_tones = [], {}, {}, {}
    native_study = None
    for alpha, registrations in trials.items():
        observed = evidence.get(alpha, {})
        if not isinstance(observed, dict) or set(observed) - set(cases):
            raise ValueError("Only registered development cases may enter this reduction")
        settled = []
        scores = {arm: [] for arm in ("adaptive", "constant")}
        statuses = {arm: [] for arm in scores}
        for case in cases:
            registration_ref = registrations[case]
            registration, _ = _read_registration(registration_ref)
            identity = _case_identity(registration)
            if (registration["scope"]["case_ref"] != case or registration["adaptive_strength"] != alpha
                    or identities.setdefault(case, identity) != identity):
                raise ValueError("Development case, alpha, or matched replay identity changed")
            scope = registration["scope"]
            shared = tuple(scope[key] for key in (
                "base_raw_sha256", "base_packet_sha256", "authored_probe_sha256", "authored_plan_sha256",
                "completed_authored_updates", "owner", "native_model_sha256", "generation_config_sha256",
                "layer_devices",
            )) + (registration["observer_binding"], registration["dose_policy"])
            if native_study is None:
                native_study = shared
            elif native_study != shared:
                raise ValueError("Development grid changed its RAW owner, authored steering or observer")
            resources = observed.get(case, {})
            firsts = resources.get("native_firsts", {})
            row = read_prefix_development_case(registration_ref, firsts, matcher=matcher,
                bank_ref=resources.get("blinded_firsts"), guard_ref=resources.get("guard_terminal"))
            for ref in firsts.values():
                first, _ = _verified_json(ref)
                _, claim = _pinned_stat(first["claim"])
                if claim.st_mtime_ns < panel_time:
                    raise ValueError("A native arm entered before the development grid was registered")
            off = row["observations"]["off"]
            if "off" in firsts:
                first, _ = _verified_json(firsts["off"])
                original = first["text"]
                content = (original["sha256"], original["bytes"])
                if off_sources.setdefault(case, content) != content:
                    raise ValueError("Development settings must preserve the exact OFF text bytes")
            if off["tone"] is not None and off_tones.setdefault(case, off["tone"]) != off["tone"]:
                raise ValueError("Development settings disagree on the original OFF intensity")
            outcomes = {}
            for arm in scores:
                on = row["observations"][arm]
                if off["tone"] is not None and on["tone"] is not None:
                    scores[arm].append((case, off["tone"], on["tone"]))
                status, reason = _case_status(row, arm)
                statuses[arm].append(status)
                outcomes[arm] = dict(status=status, reason=reason)
            settled.append(row | dict(outcomes=outcomes))
        summaries = {}
        for arm, values in scores.items():
            gain = matched_effect_mean(case_ids=cases, observations=values) if len(values) == len(cases) else None
            summaries[arm] = dict(status=_trial_status(statuses[arm]), mean_gain=gain)
        results.append(dict(alpha=alpha, **summaries["adaptive"], matched_constant=summaries["constant"], cases=settled))

    eligible = [trial for trial in results if trial["status"] == "pass"]
    best_mean = max((trial["mean_gain"] for trial in eligible), default=None)
    best = [trial["alpha"] for trial in eligible if trial["mean_gain"] == best_mean]
    return dict(schema="arete.a_psr.prefix_development_result.v1", development_registration=panel_ref,
                case_ids=cases, trials=results, best_alphas=best, best_mean_gain=best_mean, ema_decay=None)


def _trial_has_pending_evidence(trial):
    if trial["status"] != "unmeasured":
        return False
    for case in trial["cases"]:
        outcome = case["outcomes"]["adaptive"]
        if outcome["status"] == "unmeasured" and outcome["reason"] not in _SETTLED_TIMING_REASONS:
            return True
    return False


def freeze_prefix_development(panel_ref, evidence, output_path, *, matcher):
    """Write a first selection receipt for one unique measured development maximum."""
    result = reduce_prefix_development(panel_ref, evidence, matcher=matcher)
    if any(_trial_has_pending_evidence(trial) for trial in result["trials"]):
        return result | dict(frozen=None, freeze_reason="incomplete_development_evidence")
    best = result["best_alphas"]
    if not best:
        return result | dict(frozen=None, freeze_reason="no_eligible_development_alpha")
    if result["best_mean_gain"] <= 0:
        return result | dict(frozen=None, freeze_reason="no_positive_development_gain")
    if len(best) != 1:
        return result | dict(frozen=None, freeze_reason="tied_development_alphas")
    path = Path(output_path)
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("The frozen selection requires its exact canonical output path")
    first = _write_receipt(path, result | dict(selected_alpha=best[0]))
    return result | dict(frozen=first, selected_alpha=best[0])
