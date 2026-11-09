from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from arete_particle_sliders import binding as product
from arete_particle_sliders.backend import (
    AreteBackend, encode_chat, text_attention_targets, FULL_TARGETS, LINEAR_TARGETS)
from particle_sliders.inference import open_inference
from particle_sliders.response import open_response
from particle_sliders.lifecycle import CheckpointError, UnsupportedOperation

ROWS = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def session(**options):
    return open_response(product.config(fixture=True, fixture_width=16,
                                       steps=4, **options), rows=ROWS)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('width', [16, 32])
def test_actual_conditional_hybrid_architecture(width):
    host = AreteBackend(fixture_width=width)
    targets = text_attention_targets(host.model)
    assert len(targets) == 3 * len(LINEAR_TARGETS) + len(FULL_TARGETS)
    assert not any('visual' in n or 'mlp' in n for n in targets)
    assert host.hidden([1, 2, 3]).shape == (1, 3, width)
    assert any('visual' in n for n, _ in host.model.named_parameters())
    original = {n: m.forward for n, m in targets.items()}
    flags = [p.requires_grad for p in host.model.parameters()]
    slider = product.AreteHNDLSlider(host.model)
    try:
        with pytest.raises(ValueError, match='unwrapped'):
            product.AreteHNDLSlider(host.model)
    finally:
        slider.close()
    assert all(m.forward == original[n] for n, m in targets.items())
    assert flags == [p.requires_grad for p in host.model.parameters()]


def test_shared_updates_gradients_frozen_base_and_exact_off():
    with session() as s:
        native = s.game.backend
        base = deepcopy(native.backend.model.state_dict())
        initial = deepcopy(native.slider.state_dict())
        off = s.evaluate(strength=0)
        assert s.ema.values is None
        s.train()
        assert s.game.completed_steps == 4
        assert not torch.equal(initial['particles'], native.particles)
        equal(base, native.backend.model.state_dict())
        assert all(p.grad is None and not p.requires_grad for p in native.backend.model.parameters())
        equal(off, s.evaluate(strength=0))
        assert not torch.equal(off, s.evaluate(strength=1))
        s.game.network.zero_grad(set_to_none=True)
        native.generated(s.game.rows[0]).square().sum().backward()
        assert native.particles.grad is not None
        assert torch.count_nonzero(native.particles.grad)
        for name, parameter in native.slider.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert len({id(a._particles) for a in native.slider.adapters.values()}) == 1
        assert sum(p is native.particles for p in native.slider.parameters()) == 1
    assert all(m.forward == old for m, old, _ in native.slider._hosts)


def test_exact_checkpoint_resume_and_raw_inference(tmp_path):
    with session() as whole:
        whole.train()
        expected = whole.checkpoint()
    with session() as part:
        part.train(stop_after=2)
        part.checkpoint(tmp_path / 'resume.pt')
    with open_response(product.config(fixture=True, fixture_width=16, steps=4),
                       rows=ROWS, resume=tmp_path / 'resume.pt') as resumed:
        resumed.train()
        equal(expected, resumed.checkpoint())
        probes = {strength: resumed.evaluate(strength=strength)
                  for strength in (0., -.5, .5, 1.)}
        artifact = resumed.export_inference(selection='raw')
        with open_inference(artifact, operation='response') as inference:
            for strength, value in probes.items():
                equal(value, inference.response(ids=[1, 2], strength=strength))
        with pytest.raises(ValueError):
            resumed.export_inference(selection='ema')


def test_pretrained_training_refuses_before_loading(monkeypatch):
    monkeypatch.setattr(product, 'AreteBackend', lambda **kw: pytest.fail('allocated'))
    with pytest.raises(ValueError, match='not qualified'):
        open_response(product.config(), rows=ROWS)


@pytest.mark.parametrize('options', [dict(ema_decay=.9), dict(fixture_width=64),
    dict(dummy=False), dict(base_precision='bfloat16'), dict(seed=True),
    dict(penalty_input_domain='continuous'), dict(pg_source_profile='unreviewed'),
    dict(device='cuda')])
def test_invalid_options_refuse_before_loading(options, monkeypatch):
    monkeypatch.setattr(product, 'AreteBackend', lambda **kw: pytest.fail('allocated'))
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(product.config(fixture=True, **options), rows=ROWS)


def test_chat_uses_native_template_with_explicit_thinking():
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert messages == [{'role': 'user', 'content': 'hello'}]
            assert options == dict(tokenize=True, add_generation_prompt=True,
                                   return_dict=False, enable_thinking=False)
            return [1, 2, 3]
    assert encode_chat(Tokenizer(), [{'role': 'user', 'content': 'hello'}]) == [1, 2, 3]


def test_full_config_meta_targets_match_official_weight_index():
    """Actual full architecture on meta: no weights, CPU RAM tensors or GPU."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    root = Path(__file__).resolve().parents[1] / 'model-source'
    config = Qwen3_5Config.from_dict(json.loads((root / 'config.json').read_text()))
    with torch.device('meta'):
        model = Qwen3_5ForConditionalGeneration(config)
    targets = text_attention_targets(model)
    assert len(targets) == 48 * 5 + 16 * 4 == 304
    weights = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    assert all(name + '.weight' in weights for name in targets)
    assert all('visual' not in name for name in targets)
    assert all(module.weight.device.type == 'meta' for module in targets.values())


def test_checkpoint_rejects_changed_rows(tmp_path):
    with session() as s:
        s.train(stop_after=1)
        s.checkpoint(tmp_path / 'resume.pt')
    altered = deepcopy(ROWS)
    altered[0]['positive_ids'] = [1, 8]
    with pytest.raises((ValueError, CheckpointError)):
        with open_response(product.config(fixture=True, fixture_width=16, steps=4),
                           rows=altered, resume=tmp_path / 'resume.pt'):
            pytest.fail('accepted changed conditioning')


def test_product_source_tampering_refuses(tmp_path, monkeypatch):
    lock = json.loads((product.ROOT / 'sources.lock.json').read_text())
    lock['product_sources']['backend.py'] = '0' * 64
    (tmp_path / 'sources.lock.json').write_text(json.dumps(lock))
    monkeypatch.setattr(product, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='product/host source closure'):
        product.verify_runtime()


def test_checkpoint_recomputation_retains_scale_and_gradients():
    plain = product._native(dict(fixture=True, fixture_width=16, seed=7))
    checkpointed = product._native(dict(fixture=True, fixture_width=16, seed=7,
                                         checkpointing=True))
    try:
        # Nonzero branches make bypassing on recomputation observably incorrect.
        with torch.no_grad():
            for adapter in plain.slider.adapters.values():
                adapter.lora_up.weight.normal_(std=.03)
        checkpointed.slider.load_state_dict(plain.slider.state_dict())
        outputs, gradients = [], []
        for native in (plain, checkpointed):
            output = native._response([1, 2, 3], scale=.75)
            assert all(a.multiplier == 0 for a in native.slider.adapters.values())
            output.square().sum().backward()
            outputs.append(output.detach())
            gradients.append({n: p.grad.detach().clone()
                              for n, p in native.slider.named_parameters()})
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-5, atol=1e-6)
        assert len(checkpointed._checkpointing) == 4
    finally:
        plain.close()
        checkpointed.close()
    assert not checkpointed._checkpointing


def test_native_autoregressive_generation_keeps_raw_branch_and_restores_scale():
    from arete_particle_sliders.backend import generate_raw
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options['return_dict'] is False
            return [1, 3]
        def decode(self, ids, **options):
            return ','.join(str(int(i)) for i in ids)
    with session() as s:
        native = s.game.backend
        generation = deepcopy(native.backend.model.generation_config.to_dict())
        s.train(stop_after=2)
        result = generate_raw(native.backend.model, Tokenizer(), native.slider,
            [{'role': 'user', 'content': 'hello'}], strength=.5, max_new_tokens=3)
        assert isinstance(result, str) and result
        assert all(a.multiplier == 0 for a in native.slider.adapters.values())
        assert native.backend.model.generation_config.to_dict() == generation


def test_unstaged_pretrained_source_refuses_before_loading(tmp_path):
    from arete_particle_sliders.backend import load_staged
    with pytest.raises(ValueError, match='pinned Arete file'):
        load_staged(tmp_path)


def test_official_tokenizer_when_staged():
    import os
    directory = os.environ.get('ARETE_TOKENIZER_DIR')
    if not directory:
        pytest.skip('optional official tokenizer files are not staged')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                              trust_remote_code=False)
    messages = [{'role': 'user', 'content': 'Write one sentence about stars.'}]
    off = encode_chat(tokenizer, messages)
    on = encode_chat(tokenizer, messages, enable_thinking=True)
    assert isinstance(off, list) and all(type(i) is int for i in off)
    assert tokenizer.decode(off).endswith('<think>\n\n</think>\n\n')
    assert tokenizer.decode(on).endswith('<think>\n')
    assert off != on
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.pad_token_id == 248044


def test_shared_cli_explicit_cpu_validation():
    from particle_sliders.session_cli import main
    path = Path(__file__).resolve().parents[1] / 'examples/fixture.json'
    assert main(['train', '--config', str(path), '--profile', 'arete-cpu-fixture',
                 '--device', 'cpu', '--no-reporting', '--check']) in (None, 0)
