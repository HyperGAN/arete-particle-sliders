"""Actual Qwen3.5 architecture fixtures and explicit local-only model loading.

The released checkpoint is conditional generation, not a plain Qwen3 decoder.
Only language attention Linears are adapted; vision, MLP, embeddings, norms,
convolution, recurrent state parameters and the output head remain frozen.
"""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

MODEL_ID = 'NousResearch/Arete-27B'
MODEL_REVISION = '108f0afbdcfaa1ceeda4a635004304dfff6d4da2'
FULL_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
LINEAR_TARGETS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')


def require_transformers():
    if version('transformers') != '5.15.0':
        raise ValueError('Arete binding requires tested transformers==5.15.0')


def make_fixture(width=32):
    """Small random *real* conditional architecture, including a vision tower."""
    if type(width) is not int or width not in (16, 32):
        raise ValueError('Arete fixture width must be 16 or 32')
    require_transformers()
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(text_config=dict(
        vocab_size=256, hidden_size=width, intermediate_size=width * 2,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128, full_attention_interval=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'],
        linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, mtp_num_hidden_layers=0,
        rope_parameters=dict(rope_type='default', rope_theta=10000.,
                             partial_rotary_factor=1., mrope_section=[1, 1, 2]),
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
            num_heads=2, out_hidden_size=width, num_position_embeddings=16,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2),
        image_token_id=250, video_token_id=251, vision_start_token_id=252,
        vision_end_token_id=253)
    config._attn_implementation = 'eager'
    return Qwen3_5ForConditionalGeneration(config).eval().requires_grad_(False)


def text_attention_targets(model):
    """Validate complete hybrid layer coverage before any forward is changed."""
    require_transformers()
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention, Qwen3_5GatedDeltaNet)
    if model.config.model_type != 'qwen3_5':
        raise ValueError('expected Qwen3.5 conditional generation architecture')
    text = model.model.language_model
    targets, seen = {}, set()
    if len(text.layers) != model.config.text_config.num_hidden_layers:
        raise ValueError('Qwen3.5 text layer count differs')
    for index, layer in enumerate(text.layers):
        kind = model.config.text_config.layer_types[index]
        if kind == 'full_attention':
            attr, cls, leaves = 'self_attn', Qwen3_5Attention, FULL_TARGETS
        elif kind == 'linear_attention':
            attr, cls, leaves = 'linear_attn', Qwen3_5GatedDeltaNet, LINEAR_TARGETS
        else:
            raise ValueError('unknown Qwen3.5 attention layer type')
        attention = getattr(layer, attr, None)
        if not isinstance(attention, cls):
            raise ValueError('Qwen3.5 attention class differs from config')
        for leaf in leaves:
            child = getattr(attention, leaf, None)
            if (type(child) is not nn.Linear or id(child) in seen
                    or getattr(child.forward, '__func__', None) is not nn.Linear.forward):
                raise ValueError('expected unique, unwrapped Qwen3.5 text Linear projection')
            seen.add(id(child))
            targets[f'model.language_model.layers.{index}.{attr}.{leaf}'] = child
    if not targets:
        raise ValueError('no Arete text attention targets')
    return targets


class AreteBackend:
    """Differentiable native hidden states; no logits-sized training allocation."""
    def __init__(self, *, fixture_width=32):
        self.model = make_fixture(fixture_width)

    def hidden(self, ids):
        config = self.model.config.text_config
        if (not isinstance(ids, (tuple, list)) or not 1 <= len(ids) <= config.max_position_embeddings
                or any(type(v) is not int or not 0 <= v < config.vocab_size for v in ids)):
            raise ValueError('nonempty in-vocabulary token IDs within context required')
        inputs = torch.tensor([ids], dtype=torch.long,
                              device=self.model.model.language_model.embed_tokens.weight.device)
        return self.model.model.language_model(input_ids=inputs, use_cache=False,
                                               return_dict=True).last_hidden_state


def encode_chat(tokenizer, messages, *, enable_thinking=False):
    """Use the checkpoint's own template, with an explicit thinking choice."""
    if type(enable_thinking) is not bool:
        raise ValueError('enable_thinking must be boolean')
    return tokenizer.apply_chat_template(messages, tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def verify_staged(directory):
    """Stream-verify every pinned snapshot file before loading pretrained weights."""
    lock_path = Path(__file__).with_name('model.lock.json')
    lock = json.loads(lock_path.read_text())
    if lock['revision'] != MODEL_REVISION or lock['model'] != MODEL_ID:
        raise ValueError('Arete source lock differs')
    directory = Path(directory).resolve()
    for name, expected in lock['files'].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError('missing or wrong-sized pinned Arete file: ' + name)
        if 'sha256' in expected:
            digest, wanted = hashlib.sha256(), expected['sha256']
        else:
            digest, wanted = hashlib.sha1(), expected['git_blob_sha1']
            digest.update(b'blob ' + str(expected['size']).encode() + b'\0')
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != wanted:
            raise ValueError('pinned Arete file hash differs: ' + name)
    return dict(model=MODEL_ID, revision=MODEL_REVISION,
                lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                files=len(lock['files']))


def load_staged(directory, *, device='cpu', dtype=torch.bfloat16):
    """Load an already verified full snapshot, with no downloads or device map.

    Caller owns current scheduler/resource admission. All source files are checked.
    This function is not used by the fixture-qualified training entry point.
    """
    require_transformers()
    verify_staged(directory)
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    directory = Path(directory).resolve()
    config = json.loads((directory / 'config.json').read_text())
    if (config.get('architectures') != ['Qwen3_5ForConditionalGeneration']
            or config.get('model_type') != 'qwen3_5'
            or config['text_config'].get('hidden_size') != 5120
            or config['text_config'].get('num_hidden_layers') != 64):
        raise ValueError('staged model differs from the pinned Arete architecture')
    # Explicit local_files_only overrides the checkpoint tokenizer's local flags.
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                              trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(directory,
        local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation='eager').to(device).eval().requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def generate_raw(model, tokenizer, slider, messages, *, strength=1.,
                 enable_thinking=False, max_new_tokens=128, do_sample=False):
    """Native AR generation with the current RAW branch active for every token.

    Preserve the checkpoint's generation configuration, including both native
    stop IDs. Signed strengths are mechanical controls, not calibrated effects.
    """
    from copy import deepcopy
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError('max_new_tokens must be in [1,4096]')
    if type(do_sample) is not bool:
        raise ValueError('do_sample must be boolean')
    ids = encode_chat(tokenizer, messages, enable_thinking=enable_thinking)
    device = model.model.language_model.embed_tokens.weight.device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    generation = deepcopy(model.generation_config)
    generation.do_sample = do_sample
    generation.max_new_tokens = max_new_tokens
    with slider.scaled(strength):
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                generation_config=generation, use_cache=True)
    return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
