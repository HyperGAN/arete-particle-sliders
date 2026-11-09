"""Full native Arete coordinates; shared core owns every PG update and packet."""
from copy import deepcopy
import torch
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from particle_sliders.lifecycle import Binding,Capabilities,read_packet
from particle_sliders.pg095 import ConditionalPG095,engine_options,validate_pg09_domain
from particle_sliders.pg09_audio import AudioResponseCritic,audio_conditions
from particle_sliders.response import ResponseBackend,ResponseCapabilities
from particle_sliders.restoration import _restore_all
from .contracts import (BACKEND,DEFAULTS,INITIALIZATION_POLICY,PROTOCOL,SOURCE_PROFILE,WEIGHT_POLICY,
    architecture,digest,rate_overrides,validate_native_options,validate_installed_core)
from .initialization import Construction,construction_for,initialize_owned,validate_receipt
from .native import NativeRuntime,NativeResponse
from .data import read_rows


def capabilities(options):
    return ResponseCapabilities('routed_particle_gan','text','native_final_AR_token_hidden','continuous_vector',
        (1,architecture(options)['response_width']),'0.9.5','release-0.9.5',penalty_input_domain='continuous',
        base_precisions=('float32','bfloat16'),qualification='New Source; actual native tiny/pretrained qualification not yet run')


TEXT_CAPABILITIES=Capabilities('routed_particle_gan',render=True,render_outputs=('text',),
    operations=('response','render'),qualification='Native full conditional-model generation Source; pretrained quality unmeasured')


def validate_training(options):
    validate_native_options(options,training=True);validate_installed_core()
    validate_pg09_domain((1,architecture(options)['response_width']),response_domain='continuous_vector',
        **engine_options(options))


def preflight(options,request,staged):
    construction=staged.get('construction')
    if set(staged)!={'construction'}or not isinstance(construction,Construction)or construction.options_sha256!=digest(options)or construction.mode not in('fresh','resume'):
        raise ValueError('use open_training with explicit fresh/resume intent before allocation/optimizers')
    if request['native']:raise ValueError('native training accepts its matched file contexts only')


def load_runtime(options):return NativeRuntime(options)


def prepare_training(options,*,construction=None):
    validate_training(options)
    if not isinstance(construction,Construction)or construction.options_sha256!=digest(options)or construction.mode not in('fresh','resume'):
        raise ValueError('fresh/resume intent required before native allocation and optimizers')
    placement=options['placement']
    devices=[0,1]if placement['kind']=='explicit_two_cuda'else([]if placement['device']=='cpu'else[int(placement['device'].split(':')[1])])
    runtime=None;native=None
    try:
        # Bind constructor RMS/zero-up laws to seed7 (or the literal cohort seed),
        # not caller RNG, before the explicit fresh helper and optimizers.
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),devices)
            runtime=load_runtime(options)
            native=NativeResponse(runtime,checkpointing=options.get('checkpointing',False))
        rows=[native.prepare(row)for row in read_rows(options['training_file'],runtime.tokenizer,
            max_tokens=options.get('max_tokens',4096),enable_thinking=options.get('enable_thinking',False))]
        cond=audio_conditions(rows);width=native.model.config.hidden_size
        if width!=architecture(options)['response_width']:raise ValueError('complete actual native hidden width differs')
        devices=[]if runtime.trainable_device.type=='cpu'else[runtime.trainable_device.index]
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),devices)
            critic=AudioResponseCritic(width,cond.shape[1]).to(runtime.trainable_device)
            receipt=initialize_owned(native.slider,critic,construction,options)
            native.fresh_initialization=receipt.as_dict()
        if any(p.device!=runtime.trainable_device or p.dtype!=torch.float32 for p in native.network.parameters()):
            raise ValueError('all original adapter Parameters must stay together on the declared PGdevice')
        game=ConditionalPG095(native,rows,critic,conditions=cond,particles=native.network.particles,
            response_domain='continuous_vector',total_steps=options.get('steps',4),batch_size=options.get('batch',2),
            seed=options.get('seed',7),recipe_overrides=rate_overrides(options),**engine_options(options))
        spec=INFERENCE.specification(native,model_identity=native.model_identity,protocol=PROTOCOL,
            configuration=dict(architecture=architecture(options),source_profile=SOURCE_PROFILE,
                initialization_policy=INITIALIZATION_POLICY,weight_policy=WEIGHT_POLICY,initialization=receipt.as_dict()))
        return Binding(game,native.close,inference=spec)
    except BaseException as primary:
        _restore_all([native.close]if native is not None else[runtime.close]if runtime is not None else[],primary=primary);raise


def validate_inference(options,spec):
    validate_installed_core();config=spec.configuration
    if set(options)-{'model_dir','placement'}:raise ValueError('RAW inference accepts local model and explicit placement only; no teacher/cache')
    if (set(config)!={'architecture','source_profile','initialization_policy','weight_policy','initialization'}
        or config['source_profile']!=SOURCE_PROFILE or config['initialization_policy']!=INITIALIZATION_POLICY
        or config['weight_policy']!=WEIGHT_POLICY or spec.identity['protocol']!=PROTOCOL):
        raise ValueError('requires the exact new Arete095 RAW cohort; no historical packet migration')
    validate_receipt(config['initialization'],cohort=config['initialization'].get('cohort_id'))
    validate_native_options(dict(options,**config['architecture']))
    validate_pg09_domain((1,config['architecture']['response_width']),response_domain='continuous_vector',
        pg_source_profile='release-0.9.5',penalty_input_domain='continuous')


def preflight_inference(options,request,spec):
    if request['operation']=='render':
        from .finished_text import validate_request
        validate_request(strength=request['strength'],output_type=request['output_type'],**request['native'])


def prepare_inference(options,spec):
    validate_inference(options,spec);runtime=load_runtime(dict(options,**spec.configuration['architecture']));native=None
    try:
        native=NativeResponse(runtime,checkpointing=False)
        # Shape construction only. Restore receipt metadata, never call helper.
        native.fresh_initialization=spec.configuration['initialization']
        def render(**request):
            from .finished_text import generate
            return generate(native,**request)
        return InferenceBinding(native.network,native_identity(native,model_identity=native.model_identity,protocol=PROTOCOL),
            native.close,response=native.response,render=render)
    except BaseException as primary:_restore_all([native.close if native is not None else runtime.close],primary=primary);raise


TRAINING=ResponseBackend(BACKEND,capabilities(DEFAULTS),prepare_training,validate_training,
    configure_capabilities=capabilities,preflight=preflight)
INFERENCE=InferenceBackend(BACKEND,TEXT_CAPABILITIES,prepare_inference,validate_inference,preflight=preflight_inference)


def open_training(config,*,resume=None,**request):
    validate_training(config.options)
    packet=deepcopy(resume)if isinstance(resume,dict)else read_packet(resume)if resume is not None else None
    construction=construction_for(config,packet)
    session=TRAINING.open(config,operation='resume'if packet is not None else'train',construction=construction,**request)
    try:
        if packet is not None:session.restore(packet)
        return session
    except BaseException as primary:_restore_all([session.close],primary=primary);raise
