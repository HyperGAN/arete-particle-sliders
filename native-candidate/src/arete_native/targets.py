"""Exact hybrid language targets, including official non-offloading dispatch hooks.

The architecture walk derives from the preserved host_source.text_attention_targets.
Only the forward-ownership predicate is extended for native Accelerate placement;
no arbitrary pre-existing adapter/wrapper is accepted or temporarily removed.
"""
from functools import partial
from torch import nn
from .host_source import require_transformers,FULL_TARGETS,LINEAR_TARGETS


def native_linear_forward(module):
    if type(module)is not nn.Linear:return False
    if not hasattr(module,'_hf_hook'):
        return getattr(module.forward,'__func__',None)is nn.Linear.forward
    from accelerate.hooks import AlignDevicesHook
    hook=module._hf_hook;forward=module.forward;old=getattr(module,'_old_forward',None)
    return(type(hook)is AlignDevicesHook and hook.offload is False and hook.no_grad is False
        and getattr(old,'__func__',None)is nn.Linear.forward
        and isinstance(forward,partial)and forward.args==(module,)and not forward.keywords
        and forward.func.__module__=='accelerate.hooks'and forward.func.__name__=='new_forward'
        and getattr(forward,'__wrapped__',None)==old
        and str(hook.execution_device)==str(module.weight.device))


def native_attention_targets(model):
    require_transformers()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention,Qwen3_5GatedDeltaNet
    if model.config.model_type!='qwen3_5':raise ValueError('actual Qwen3.5 conditional architecture required')
    text=model.model.language_model;targets={};seen=set()
    if len(text.layers)!=model.config.text_config.num_hidden_layers:raise ValueError('native text layer count differs')
    for index,layer in enumerate(text.layers):
        kind=model.config.text_config.layer_types[index]
        if kind=='full_attention':attr,cls,leaves='self_attn',Qwen3_5Attention,FULL_TARGETS
        elif kind=='linear_attention':attr,cls,leaves='linear_attn',Qwen3_5GatedDeltaNet,LINEAR_TARGETS
        else:raise ValueError('unknown native hybrid attention type')
        attention=getattr(layer,attr,None)
        if not isinstance(attention,cls):raise ValueError('native attention class differs from config')
        for leaf in leaves:
            child=getattr(attention,leaf,None)
            if not native_linear_forward(child)or id(child)in seen:raise ValueError('unique native Linear or exact official dispatch wrapper required')
            seen.add(id(child));targets[f'model.language_model.layers.{index}.{attr}.{leaf}']=child
    if not targets:raise ValueError('no native attention targets')
    return targets
