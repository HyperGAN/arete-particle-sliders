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
from particle_sliders.pg095 import validate_pg09_domain
from arete_native95 import contracts as c,lifecycle as current
from arete_native95 import initialization as initialization
from arete_native95.host_source import make_fixture
from arete_native95.finished_text import generate


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
        pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous',
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
    gate=validate_pg09_domain((1,5120),response_domain='continuous_vector',pg_source_profile='release-0.9.5',penalty_input_domain='continuous')
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
                pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous',ema_decay=decay))


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


def test_real_official_dispatch_hooks_preserved_and_gradients(setup):
    from accelerate.hooks import attach_execution_device_hook
    from arete_native95.targets import native_attention_targets
    from arete_native95.native import NativeResponse
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
