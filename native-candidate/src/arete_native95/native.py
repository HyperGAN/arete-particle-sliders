"""Exact local Qwen3.5 conditional checkpoint; explicit frozen-model placement."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from torch import nn
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter
from particle_sliders.restoration import _restore_all
from . import host_source as host
from .contracts import MODEL,REVISION,architecture,file_hash,validate_placement
from .slider import AreteSlider
from .targets import native_attention_targets


class NativeRuntime:
    def __init__(self,options):
        host.require_transformers()
        self.closed=False;self.model=None;self.tokenizer=None
        root=Path(options['model_dir']).resolve();verified=host.verify_staged(root)
        config=json.loads((root/'config.json').read_text());arch=architecture(options)
        if (config.get('architectures')!=['Qwen3_5ForConditionalGeneration']or config.get('model_type')!='qwen3_5'
            or config['text_config']['hidden_size']!=5120 or config['text_config']['num_hidden_layers']!=64
            or config['text_config']['layer_types']!=(['linear_attention']*3+['full_attention'])*16):
            raise ValueError('exact full native Arete27B hybrid5120 architecture required')
        if arch['response_width']!=5120:raise ValueError('no native response projection/truncation')
        self.placement=deepcopy(validate_placement(options['placement'],config))
        self.trainable_device=torch.device(self.placement['trainable_device'])
        self.dtype=getattr(torch,arch['base_precision'])
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        self.tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        kwargs=dict(local_files_only=True,trust_remote_code=False,dtype=self.dtype,attn_implementation='eager',
            low_cpu_mem_usage=True,use_safetensors=True,output_loading_info=True)
        if self.placement['kind']=='explicit_two_cuda':
            if torch.cuda.device_count()!=2:raise ValueError('explicit two-device plan requires exactly two owned visible GPUs')
            kwargs['device_map']=self.placement['device_map']
        try:
            self.model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(str(root),**kwargs)
            if any(loading.get(key)for key in('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise ValueError('entire pretrained conditional model must load without random replacement')
            if self.placement['kind']=='single':self.model.to(self.placement['device'])
            else:
                actual={key:str(value)if not isinstance(value,int)else'cuda:'+str(value)
                    for key,value in self.model.hf_device_map.items()}
                if actual!=self.placement['device_map']:raise ValueError('actual native device map differs from explicit plan')
            self.model.eval().requires_grad_(False)
            for name,parameter in self.model.named_parameters():
                if self.placement['kind']=='single':expected=self.placement['device']
                else:
                    matches=[key for key in self.placement['device_map']if name==key or name.startswith(key+'.')]
                    if not matches:raise ValueError('unmapped original native parameter:'+name)
                    expected=self.placement['device_map'][max(matches,key=len)]
                if parameter.is_meta or parameter.device!=torch.device(expected)or parameter.dtype!=self.dtype:
                    raise ValueError('actual complete unquantized parameter placement/dtype differs:'+name)
            if len(native_attention_targets(self.model))!=304:raise ValueError('all304 native hybrid attention projections required')
            self.input_device=self.model.model.language_model.embed_tokens.weight.device
            self.identity=dict(model=MODEL,revision=REVISION,verified_snapshot=verified,
                architecture=config,selection='raw',base_precision=arch['base_precision'])
        except BaseException:
            self.close();raise

    def close(self):
        self.model=None;self.tokenizer=None;self.closed=True


class _LanguageHost(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.config=model.config.text_config


class NativeResponse(ARLastHiddenResponseAdapter):
    family='arete095_full_native_final_token'
    def __init__(self,runtime,*,checkpointing=False):
        self.runtime=runtime;self.native_model=runtime.model;self.tokenizer=runtime.tokenizer
        self.slider=AreteSlider(runtime.model,trainable_device=runtime.trainable_device)
        identity=deepcopy(runtime.identity)
        identity['native_product_sources']={p.name:file_hash(p)for p in sorted(Path(__file__).parent.iterdir())
            if p.is_file()and p.suffix in('.py','.json')}
        self.fresh_initialization=None;self._native_closed=False
        try:
            super().__init__(_LanguageHost(runtime.model),self.slider,model_identity=identity,
                checkpoint_modules=tuple(runtime.model.model.language_model.layers)if checkpointing else(),
                pg_source_profile='release-0.9.5',penalty_input_domain='continuous')
        except BaseException as primary:_restore_all([self.slider.close,runtime.close],primary=primary);raise

    def _hidden(self,ids):
        config=self.native_model.config.text_config
        if (not ids or len(ids)>config.max_position_embeddings
            or any(type(value)is not int or not 0<=value<config.vocab_size for value in ids)):
            raise ValueError('entire in-vocabulary native context required; no truncation')
        input_ids=torch.tensor([ids],dtype=torch.long,device=self.runtime.input_device)
        output=self.native_model.model.language_model(input_ids=input_ids,use_cache=False,return_dict=True)
        # Same native5120 coordinates. Only a differentiable device copy, never
        # dimension slicing/projection; the shared game requires PGdevice output.
        return output.last_hidden_state[:,-1].to(device=self.runtime.trainable_device)

    @torch.no_grad()
    def response(self,*,ids=None,messages=None,enable_thinking=False,max_tokens=4096,strength=1.,seed=0):
        from .data import encode_prompt
        if (ids is None)==(messages is None):raise ValueError('exactly one complete ids/messages context required')
        ids=encode_prompt(self.tokenizer,messages,max_tokens=max_tokens,enable_thinking=enable_thinking)if messages is not None else ids
        if len(ids)>max_tokens:raise ValueError('native context too long; no truncation')
        return self._response(tuple(ids),scale=strength).detach().clone()

    def metadata(self):
        from .contracts import SOURCE_PROFILE,INITIALIZATION_POLICY,WEIGHT_POLICY
        return dict(super().metadata(),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,
            fresh_initialization=self.fresh_initialization,weight_policy=WEIGHT_POLICY,
            placement=deepcopy(self.runtime.placement),native_full_width=self.native_model.config.text_config.hidden_size)

    def close(self):
        if self._native_closed:return
        _restore_all([self.close_checkpointing,self.slider.close,self.runtime.close])
        # Owned base release only after all original forwards/flags restore.
        self.model.model=None;self.native_model=None;self.tokenizer=None;self._native_closed=True
