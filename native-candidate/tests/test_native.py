"""Authored native tiny CPU conformance, not run by the Source author.

No pretrained tensor/network calls are permitted. Root runs these in a new
bounded exact current-core/Transformers5.15 environment, never an old prefix.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.inference import open_inference
from particle_sliders.engine import validate_engine_domain
from arete_native import contracts as c,lifecycle as current
from arete_native import initialization as initialization
from arete_native.host_source import make_fixture
from arete_native.finished_text import generate
from arete_native_scheduling import generate as generate_scheduled


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.fixture(autouse=True)
def no_pretrained_or_network(monkeypatch):
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    def blocked(*args,**kwargs):pytest.fail('pretrained/network call in native tiny test')
    monkeypatch.setattr(Qwen3_5ForConditionalGeneration,'from_pretrained',blocked)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',blocked)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'model';root.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7);example=make_fixture(16)
    config=example.config.to_dict();(root/'config.json').write_text(json.dumps(config));del example
    rows=tmp_path/'rows.jsonl'
    rows.write_text('\n'.join(json.dumps(x)for x in[
        dict(neutral_ids=[1,3],positive_ids=[1,4]),dict(neutral_ids=[1,5],positive_ids=[1,6])])+'\n')
    opts=dict(model_dir=str(root),placement=c.single_placement(),training_file=str(rows),response_width=16,base_precision='float32',
        steps=2,batch=2,seed=7,ema_decay=None,source_profile=c.SOURCE_PROFILE,initialization_policy=c.INITIALIZATION_POLICY,
        pg_version='0.11.1',pg_source_profile='release-0.11.1',penalty_input_domain='continuous',
        lr=.001,d_lr_mult=18.,prior_lr_mult=30.)
    runtimes=[]
    def tiny_runtime(options):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7);model=make_fixture(16)
        runtime=SimpleNamespace(model=model,tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:' '.join(map(str,ids))),
            trainable_device=torch.device('cpu'),input_device=torch.device('cpu'),dtype=torch.float32,placement=c.single_placement(),
            identity=dict(qualification='SYNTHETIC random real Qwen3.5 architecture only; NOT pretrained',architecture=model.config.to_dict()))
        runtime.close=lambda:None
        runtimes.append(runtime);return runtime
    monkeypatch.setattr(current,'load_runtime',tiny_runtime)
    return SessionConfig(c.BACKEND,'synthetic-current-native-v1',**opts),runtimes


def test_current_core_and_full5120_continuous_gate():
    c.validate_installed_core()
    gate=validate_engine_domain((1,5120),response_domain='continuous_vector',pg_source_profile='release-0.11.1',penalty_input_domain='continuous')
    assert gate['penalty_input_domain']=='continuous'


def test_literal_rates_and_ema_refusal_before_any_path_access():
    values=dict(lr=.001,d_lr_mult=18.,prior_lr_mult=30.)
    assert c.rate_overrides(values)==values
    for name in values:
        for value in(True,False,0,-1,float('inf'),float('nan'),'0.001'):
            with pytest.raises(ValueError):c.rate_overrides({name:value})
    for decay in(.995,0.,False):
        with pytest.raises(ValueError,match='EMA'):
            c.validate_profile(dict(source_profile=c.SOURCE_PROFILE,initialization_policy=c.INITIALIZATION_POLICY,
                pg_version='0.11.1',pg_source_profile='release-0.11.1',penalty_input_domain='continuous',ema_decay=decay))


def test_two_device_map_covers_every_official_native_parameter_boundary():
    root=Path(__file__).parents[2]/'model-source'
    config=json.loads((root/'config.json').read_text());index=json.loads((root/'model.safetensors.index.json').read_text())
    placement=c.two_device_placement();assert c.validate_placement(placement,config)==placement
    for name in index['weight_map']:
        matches=[key for key in placement['device_map']if name.startswith(key+'.')]
        assert len(matches)==1
    assert placement['device_map']['model.language_model.layers.31']=='cuda:0'
    assert placement['device_map']['model.language_model.layers.32']=='cuda:1'
    bad=deepcopy(placement);bad['device_map']['model.language_model.layers.32']='cpu'
    with pytest.raises(ValueError):c.validate_placement(bad,config)
    assert not torch.cuda.is_initialized()


def test_actual_tiny_gradient_original_parameters_zero_up_and_unit_bank(setup):
    config,runtimes=setup
    with current.open_training(config)as session:
        backend=session.game.backend;slider=backend.slider
        assert len(slider.target_names)==19
        assert all(projection._particles is slider.particles for projection in slider.projections)
        assert all(torch.count_nonzero(p.lora_up.weight)==0 and torch.count_nonzero(p.lora_down.weight)>0 for p in slider.projections)
        assert {id(p)for p in session.game.network.parameters()}=={id(p)for p in slider.parameters()}
        assert all(not p.requires_grad for p in backend.native_model.parameters())
        receipt=backend.fresh_initialization
        assert receipt['components']['adapter.bank']['parameters']['particles']['host_declaration']['std']==1.
        assert [group['lr']for group in session.game.opt_g.param_groups]==[.001,.03]
        assert all(group['lr']==.018000000000000002 for group in session.game.opt_d.param_groups)
        session.train(stop_after=1)
        assert session.game.completed_steps==1 and session.ema.values is None
        assert any(torch.count_nonzero(p.lora_up.weight)>0 for p in slider.projections)
        assert all(p.grad is None for p in backend.native_model.parameters())


@pytest.mark.parametrize('checkpointing',[False,True])
def test_exact_full_resume_raw_no_initializer_or_teacher_on_inference(setup,monkeypatch,checkpointing):
    config,runtimes=setup
    config=SessionConfig(config.backend,config.cohort,**dict(config.options,checkpointing=checkpointing))
    with current.open_training(config)as full:full.train();expected=full.checkpoint()
    with current.open_training(config)as split:split.train(stop_after=1);packet=split.checkpoint()
    monkeypatch.setattr(initialization,'initialize_fresh_components',lambda *a,**kw:pytest.fail('fresh initialization on restore/RAW'))
    with current.open_training(config,resume=packet)as resumed:
        assert resumed.game.completed_steps==1
        resumed.train();equal(expected,resumed.checkpoint());artifact=resumed.export_inference(selection='raw')
    monkeypatch.setattr(current,'read_rows',lambda *a,**kw:pytest.fail('teacher file read by RAW inference'))
    local=dict(model_dir=config.options['model_dir'],placement=c.single_placement())
    with open_inference(artifact,backend=current.INFERENCE,operation='response',options=local)as inference:
        response=inference.response(ids=[1,3],strength=.5)
        assert response.shape==(1,16)and inference.selection=='raw'
    assert not torch.cuda.is_initialized()


def test_actual_tiny_native_cached_generation_and_rng_preservation(setup,tmp_path,monkeypatch):
    config,runtimes=setup
    with current.open_training(config)as session:
        native=session.game.backend;rng=torch.get_rng_state().clone();state=deepcopy(native.native_model.state_dict())
        # Exercise the actual TF5.15 resolution path with hostile inherited
        # sampled/cache/length defaults; preserve the checkpoint's native IDs.
        model=native.native_model
        for key,value in dict(do_sample=True,num_beams=4,num_return_sequences=2,use_cache=False,
                max_new_tokens=17,min_new_tokens=7,min_length=9,temperature=.1,top_k=2,top_p=.25,
                return_dict_in_generate=False,output_scores=True).items():
            setattr(model.generation_config,key,value)
        inherited=deepcopy(model.generation_config.to_dict());resolved=[]
        prepare=model._prepare_generation_config
        def capture(generation_config,**kwargs):
            assert generation_config is not model.generation_config
            assert set(kwargs)<={'input_ids','attention_mask'}
            actual,remaining=prepare(generation_config,**kwargs)
            resolved.append(deepcopy(actual.to_dict()))
            return actual,remaining
        monkeypatch.setattr(model,'_prepare_generation_config',capture)
        off=generate(native,ids=[1,3],strength=0,max_new_tokens=2,seed=7)
        on=generate(native,ids=[1,3],strength=1,max_new_tokens=2,seed=7)
        assert off.text==on.text
        assert off.native['actual_generate_kwargs']==dict(generation_config='fresh_per_call_explicit',model_inputs=['input_ids','attention_mask'])
        assert len(resolved)==2
        for actual in resolved:
            assert actual['do_sample']is False and actual['use_cache']is True and actual['max_new_tokens']==2
            assert actual['num_beams']==actual['num_return_sequences']==1
            assert actual['min_length']==actual['min_new_tokens']==0
            assert actual['temperature']==actual['top_p']==1. and actual['top_k']==50
            assert actual['return_dict_in_generate']is True and actual['output_scores']is False
            for key in('eos_token_id','bos_token_id','pad_token_id'):
                assert actual[key]==inherited[key]==off.native['explicit_generation_config'][key]
        assert model.generation_config.to_dict()==inherited
        assert torch.equal(rng,torch.get_rng_state())
        equal(state,native.native_model.state_dict())
        path=tmp_path/'native.txt';off.save(path);assert path.read_text()==off.text
        with pytest.raises(ValueError,match='fresh'):off.save(path)
        assert not torch.cuda.is_initialized()


def test_missing_intent_and_legacy_backend_refused_before_allocation(setup,monkeypatch):
    config,_=setup
    monkeypatch.setattr(current,'load_runtime',lambda *a:pytest.fail('native allocation before refusal'))
    with pytest.raises(ValueError,match='intent'):current.TRAINING.open(config)
    old=SessionConfig('arete-particle-sliders','old-fixture',**config.options)
    with pytest.raises(ValueError,match='historical'):current.open_training(old)


def test_scheduled_generation_preserves_the_strict_raw_renderer(setup):
    config, runtimes = setup
    with current.open_training(config) as training:
        training.train(stop_after=1)
        artifact = training.export_inference(selection='raw')
    options = dict(model_dir=config.options['model_dir'], placement=c.single_placement())
    request = dict(ids=[1, 3], max_new_tokens=2)
    with open_inference(artifact, backend=current.INFERENCE, options=options,
                        operation='render', output_type='text', native_request=request) as session:
        native = session.binding.close.__self__
        applied = []

        def force_token(_layer, _args, logits):
            applied.append(native.network.scale)
            result = logits.clone()
            result[..., 5] += 100
            return result

        with runtimes[-1].model.lm_head.register_forward_hook(force_token):
            original = session.render(strength=0.5, **request)
            constant = generate_scheduled(session, step_gains=[1, 1], strength=0.5, **request)
            varied = generate_scheduled(session, step_gains=[1.5, 0.5], strength=0.5, **request)
            final = session.render(strength=0.5, **request)
        assert original.text == constant.text == final.text
        assert original.native == {k: v for k, v in constant.native.items() if k != 'strength_schedule'}
        assert final == original
        assert varied.native['strength_schedule'] == dict(prefix_tokens=2, effective_strengths=[0.75, 0.25])
        assert applied == [0.5, 0.5, 0.5, 0.5, 0.75, 0.25, 0.5, 0.5]


def test_real_official_dispatch_hooks_preserved_and_gradients(setup):
    from accelerate.hooks import attach_execution_device_hook
    from arete_native.targets import native_attention_targets
    from arete_native.native import NativeResponse
    config,runtimes=setup
    runtime=current.load_runtime(config.options)
    attach_execution_device_hook(runtime.model,torch.device('cpu'))
    targets=native_attention_targets(runtime.model);before={name:(module.forward,module._hf_hook)for name,module in targets.items()}
    native=NativeResponse(runtime)
    try:
        value=native._response((1,3),scale=1.)
        assert value.shape==(1,16)
        value.square().mean().backward()
        assert any(p.grad is not None for p in native.network.parameters())
        assert all(p.grad is None for p in runtime.model.parameters())
    finally:native.close()
    for name,module in native_attention_targets(runtime.model).items():
        assert module.forward==before[name][0]and module._hf_hook is before[name][1]
    assert not torch.cuda.is_initialized()


@pytest.fixture
def adaptive_setup(setup, monkeypatch):
    import os
    import sys
    from arete_native_scheduling import load_steering_core

    wheel = os.getenv("ARETE_STEERING_CORE_WHEEL")
    if wheel is None:
        pytest.skip("Set ARETE_STEERING_CORE_WHEEL to the shared b1f780d wheel")
    import particle_sliders

    monkeypatch.delenv("ARETE_STEERING_CORE_WHEEL")
    owners = particle_sliders.__file__, tuple(particle_sliders.__path__), tuple(sys.path)
    rng = torch.random.get_rng_state()
    core = load_steering_core(wheel)
    assert core.adaptive.__name__.startswith("arete_native_scheduling.")
    assert core.prompt.__name__.startswith("arete_native_scheduling.")
    assert owners == (particle_sliders.__file__, tuple(particle_sliders.__path__), tuple(sys.path))
    assert torch.equal(rng, torch.random.get_rng_state())
    c.validate_installed_core()

    config, runtimes = setup
    with current.open_training(config) as training:
        training.train(stop_after=1)
        artifact = training.export_inference(selection="raw")
    options = dict(model_dir=config.options["model_dir"], placement=c.single_placement())
    request = dict(ids=[1, 3], max_new_tokens=4)
    with open_inference(artifact, backend=current.INFERENCE, options=options,
                        operation="render", output_type="text", native_request=request) as session:
        yield session, runtimes[-1].model, core, request


def test_adaptive_cfp_raw_exact_logits_at_eos(adaptive_setup):
    from arete_native_scheduling import CachedGate

    session, model, core, request = adaptive_setup
    original_scale = session.binding.network.scale
    observed = []

    def stop_after_prefill(_layer, _args, logits):
        observed.append(logits.detach().clone())
        result = logits.clone()
        token = 5 if len(observed) == 1 else model.generation_config.eos_token_id
        result[..., token] += 100
        return result

    direction = torch.ones(16)

    def config(threshold):
        return CachedGate(core=core, layer=0, method="cfp", parameter=threshold,
                          intervention="raw_adapter", direction=direction, evidence_ref="test fixture")

    with model.lm_head.register_forward_hook(stop_after_prefill):
        off = session.render(strength=0.0, **request)
        off_logits = observed[:]
        observed.clear()
        never = generate_scheduled(session, steering=config(-1e20), strength=0.5, **request)
        equal(observed, off_logits)
        observed.clear()
        disabled = generate_scheduled(session, steering=config(1e20), strength=0.0, **request)
        equal(observed, off_logits)
        observed.clear()
        always = generate_scheduled(session, steering=config(1e20), strength=0.5, **request)
        always_logits = observed[:]
        observed.clear()
        fixed = generate_scheduled(session, step_gains=[0, 1, 1, 1], strength=0.5, **request)
        equal(observed, always_logits)

    assert off.text == never.text == disabled.text
    assert always.text == fixed.text
    report = always.native["adaptive_steering"]
    assert report["usage"] == {"fired": 1, "eligible": 1}
    assert report["effective_strengths"] == [0.0, 0.5]
    assert report["cached_steps"][0]["cache_position"] == 2
    assert disabled.native["adaptive_steering"]["usage"] == {"fired": 0, "eligible": 0}
    assert session.binding.network.scale == original_scale
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not model.model.language_model.layers[0]._forward_pre_hooks


def test_adaptive_clabs_and_psr_edit_native_different_layers(adaptive_setup):
    from arete_native_scheduling import CachedGate, CachedPromptSteering

    session, model, core, request = adaptive_setup

    def continue_native(_layer, _args, logits):
        result = logits.clone()
        result[..., 5] += 100
        return result

    direction = torch.arange(1.0, 17.0)
    clabs = core.adaptive.CLABSParameters(mean=0, std=1, epsilon=0.001,
        setpoint=0.7, proportional_gain=0.3, integral_gain=0.2, beta=0.4)
    policy = CachedGate(core=core, layer=2, method="clabs", parameter=0.5,
                        intervention="residual", seed=7, direction=direction,
                        clabs=clabs, evidence_ref="test fixture")
    fit = core.prompt.PromptSteering(16)
    with torch.no_grad():
        fit.probe.weight.zero_()
        fit.probe.bias.fill_(0.25)
        fit.direction.weight.fill_(0.5)
    psr = CachedPromptSteering(core=core, layer=3, fitted=fit, evidence_ref="test fit")
    with model.lm_head.register_forward_hook(continue_native):
        first = generate_scheduled(session, steering=policy, strength=0.7, **request)
        second = generate_scheduled(session, steering=policy, strength=0.7, **request)
        observed = []
        layer = model.model.language_model.layers[3]
        with layer.input_layernorm.register_forward_pre_hook(
                lambda _layer, args: observed.append(args[0].detach().clone())):
            session.render(strength=0.0, **request)
            unedited, observed[:] = observed[:], []
            learned = generate_scheduled(session, steering=psr, strength=-0.5, **request)
        assert torch.equal(unedited[0], observed[0])
        for before, after in zip(unedited[1:], observed[1:], strict=True):
            torch.testing.assert_close(after, before - 0.0625, rtol=0, atol=0)

    assert first == second
    report = first.native["adaptive_steering"]
    expected = core.adaptive.ActivationGate("clabs", 0.5, seed=7, clabs=clabs)
    for step in report["cached_steps"]:
        decision = expected.step(step["alignment"])
        assert (step["probability"], step["fired"]) == (decision.probability, decision.fired)
    assert report["usage"] == {"fired": expected.usage.fired, "eligible": 3}
    assert learned.native["adaptive_steering"]["effective_strengths"] == [0, -0.125, -0.125, -0.125]
    assert [step["coefficient"] for step in learned.native["adaptive_steering"]["cached_steps"]] == [0.25] * 3
    assert fit.training and all(parameter.requires_grad for parameter in fit.parameters())

    def fail_cached(_layer, _args, _logits):
        if session.binding.network.scale:
            raise RuntimeError("test failure during a gated native forward")
        return continue_native(_layer, _args, _logits)

    failing = CachedGate(core=core, layer=0, method="random", parameter=1.0,
                         intervention="raw_adapter", seed=7, evidence_ref="test fixture")
    baseline = session.render(strength=0.0, **request)
    with model.lm_head.register_forward_hook(fail_cached):
        with pytest.raises(RuntimeError, match="gated native forward"):
            generate_scheduled(session, steering=failing, strength=0.5, **request)
    assert session.render(strength=0.0, **request) == baseline
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not any(layer._forward_pre_hooks for layer in model.model.language_model.layers)


def test_post_block_psr_edits_first_answer_and_cleans_up(adaptive_setup):
    from arete_native_scheduling import PostBlockPromptSteering

    session, model, core, request = adaptive_setup
    language = model.model.language_model
    last_block = language.layers[-1]
    fit = core.prompt.PromptSteering(16)
    with torch.no_grad():
        fit.probe.weight.zero_()
        fit.probe.bias.fill_(0.25)
        fit.direction.weight.copy_(torch.arange(16).reshape(1, -1) / 16)
    policy = PostBlockPromptSteering(core=core, layer=len(language.layers) - 1,
                                   fitted=fit, evidence_ref="test fixture")
    original_scale = session.binding.network.scale
    residuals, logits = [], []

    def record_residual(_norm, args):
        residuals.append(args[0].detach().clone())

    def fixed_eos(_head, _args, output):
        logits.append(output.detach().clone())
        result = output.clone()
        token = 5 if len(logits) == 1 else model.generation_config.eos_token_id
        result[..., token] += 100
        return result

    with language.norm.register_forward_pre_hook(record_residual), model.lm_head.register_forward_hook(fixed_eos):
        baseline = session.render(strength=0.0, **request)
        base_residuals, base_logits = residuals[:], logits[:]
        residuals.clear()
        logits.clear()
        dormant = generate_scheduled(session, steering=policy, strength=0.0, **request)
        assert dormant.native["new_token_ids"] == baseline.native["new_token_ids"]
        equal(residuals, base_residuals)
        equal(logits, base_logits)
        residuals.clear()
        logits.clear()
        active = generate_scheduled(session, steering=policy, strength=0.5, **request)
        assert active.native["new_token_ids"] == baseline.native["new_token_ids"]
        assert torch.equal(residuals[0][:, :-1], base_residuals[0][:, :-1])
        assert not torch.equal(logits[0], base_logits[0])
        for actual, original in zip(residuals, base_residuals, strict=True):
            expected, _ = fit(original[:, -1:], 0.5)
            assert torch.equal(actual[:, -1:], expected)

    receipt = active.native["adaptive_steering"]
    assert receipt["decoder_output_layer"] == len(language.layers) - 1
    assert receipt["position_scope"] == "answer_only" and "decoder_input_layer" not in receipt
    assert receipt["effective_strengths"] == [0.125, 0.125]
    assert [item["output_index"] for item in receipt["answer_steps"]] == [0, 1]
    assert [item["cache_position"] for item in receipt["answer_steps"]] == [1, 2]
    assert fit.training and all(parameter.requires_grad for parameter in fit.parameters())

    def fail(_head, _args, _output):
        raise RuntimeError("fixture failure after the edited first answer residual")

    with model.lm_head.register_forward_hook(fail):
        with pytest.raises(RuntimeError, match="fixture failure"):
            generate_scheduled(session, steering=policy, strength=0.5, **request)
    assert not model._forward_hooks and not model._forward_pre_hooks and not last_block._forward_hooks
    assert session.binding.network.scale == original_scale


def test_adaptive_core_rejects_unverified_bytes(tmp_path):
    from zipfile import ZipFile

    from arete_native_scheduling import load_steering_core

    wheel = tmp_path / "changed.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("particle_sliders/adaptive_steering.py", b"X" * 10329)
    with pytest.raises(ValueError, match="source checksum differs"):
        load_steering_core(wheel)
