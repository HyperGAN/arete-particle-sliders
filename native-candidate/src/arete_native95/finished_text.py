"""Native Qwen3.5 autoregressive text, never a decoded hidden-vector substitute."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json,math,os
import torch
from .data import encode_prompt


def validate_request(*,messages=None,ids=None,enable_thinking=False,max_tokens=4096,max_new_tokens=128,
                     strength=1.,seed=0,output_type='text',images=None):
    if images is not None or output_type!='text':raise ValueError('native text-only Arete generation')
    if (messages is None)==(ids is None):raise ValueError('exactly one complete native messages/ids context required')
    if type(enable_thinking)is not bool:raise ValueError('explicit Boolean thinking choice required')
    for key,value,maximum in(('max_tokens',max_tokens,32768),('max_new_tokens',max_new_tokens,4096)):
        if type(value)is not int or not 1<=value<=maximum:raise ValueError('bounded '+key+' required')
    if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('explicit native seed required')
    if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('finite strength0..1 required')


@dataclass(frozen=True)
class TextOutput:
    text:str
    native:dict
    def save(self,path):
        path=Path(path);sidecar=Path(str(path)+'.native.json')
        if path.exists()or sidecar.exists():raise ValueError('fresh finished-text/receipt paths required')
        with path.open('x')as stream:stream.write(self.text);stream.flush();os.fsync(stream.fileno())
        with sidecar.open('x')as stream:json.dump(self.native,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


@torch.no_grad()
def generate(native,*,messages=None,ids=None,enable_thinking=False,max_tokens=4096,max_new_tokens=128,
             strength=1.,seed=0,output_type='text',images=None):
    validate_request(messages=messages,ids=ids,enable_thinking=enable_thinking,max_tokens=max_tokens,
        max_new_tokens=max_new_tokens,strength=strength,seed=seed,output_type=output_type,images=images)
    runtime=native.runtime;model=native.native_model
    if runtime.tokenizer is None:raise ValueError('actual checkpoint tokenizer is required for finished text')
    ids=encode_prompt(runtime.tokenizer,messages,max_tokens=max_tokens,enable_thinking=enable_thinking)if messages is not None else ids
    cfg=model.config.text_config
    if (not isinstance(ids,(list,tuple))or not 1<=len(ids)<=max_tokens
        or len(ids)+max_new_tokens>cfg.max_position_embeddings
        or any(type(value)is not int or not 0<=value<cfg.vocab_size for value in ids)):
        raise ValueError('entire original native context required; no truncation/vocabulary projection')
    inputs=torch.tensor([ids],dtype=torch.long,device=runtime.input_device)
    from transformers import GenerationConfig
    # Keep only the checkpoint's actual native token IDs. Own the per-call
    # generation policy explicitly; never copy its sampled/length defaults.
    tokens={name:deepcopy(getattr(model.generation_config,name))for name in
        ('eos_token_id','bos_token_id','pad_token_id')}
    for name,value in tokens.items():
        values=value if name=='eos_token_id'and isinstance(value,(list,tuple))else[value]
        if not values or any(type(token)is not int or not 0<=token<cfg.vocab_size for token in values):
            raise ValueError('explicit checkpoint native EOS/BOS/PAD IDs required: '+name)
    policy=dict(do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,
        max_new_tokens=max_new_tokens,min_new_tokens=0,min_length=0,
        temperature=1.,top_k=50,top_p=1.,repetition_penalty=1.,no_repeat_ngram_size=0,
        return_dict_in_generate=True,output_scores=False,output_logits=False,
        output_attentions=False,output_hidden_states=False,**tokens)
    generation=GenerationConfig(**policy)
    devices=sorted({parameter.device.index for parameter in model.parameters()if parameter.is_cuda}|
                   {parameter.device.index for parameter in native.network.parameters()if parameter.is_cuda})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:torch.cuda.manual_seed_all(seed)
        with native.network.scaled(strength):
            generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                generation_config=generation)
    sequences=generated.sequences
    if sequences.ndim!=2 or sequences.shape[0]!=1 or sequences.shape[1]<len(ids)or sequences[0,:len(ids)].tolist()!=list(ids):
        raise ValueError('one actual generated native sequence with unchanged complete prefix required')
    tokens=sequences[0,len(ids):].detach().cpu().tolist()
    text=runtime.tokenizer.decode(tokens,skip_special_tokens=False)
    return TextOutput(text,dict(schema='arete095-native-finished-text-v1',model=runtime.identity,
        selection='raw',weight_EMA=None,prompt_tokens=len(ids),new_token_ids=tokens,
        native_eos_token_id=generation.eos_token_id,native_pad_token_id=generation.pad_token_id,
        native_bos_token_id=generation.bos_token_id,
        enable_thinking=enable_thinking,strength=strength,seed=seed,
        actual_generate_kwargs=dict(generation_config='fresh_per_call_explicit',
            model_inputs=['input_ids','attention_mask']),explicit_generation_config=policy,
        placement=runtime.placement,pretrained_quality_qualified=False))
