"""Allocation-free new-cohort options, explicit placement and current-core pins."""
import hashlib
import json
import math
from pathlib import Path

CORE_REVISION='41f6066f2d2644902670d5eb736de052e1a70553'
BACKEND='arete_native95'
SOURCE_PROFILE='arete-native5120-pg095-explicit-placement-fresh-v1'
INITIALIZATION_POLICY='arete-kaiming-RMS-QR-zero-up-bank-R2-std1-v1'
PROTOCOL='arete095-native-final-token-5120-and-finished-text-v1'
WEIGHT_POLICY='raw-no-weight-EMA'
MODEL='NousResearch/Arete-27B'
REVISION='108f0afbdcfaa1ceeda4a635004304dfff6d4da2'
DEFAULTS=dict(rank=8,particles=128,particle_dim=4,response_width=5120,base_precision='bfloat16',checkpointing=False)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def architecture(options):return {key:options.get(key,value)for key,value in DEFAULTS.items()}


def single_placement(device='cpu'):
    if device not in('cpu','cuda:0','cuda:1'):raise ValueError('explicit supported single device required')
    return dict(kind='single',device=device,trainable_device=device)


def two_device_placement(*,layers=64,first_layers=32):
    """Complete explicit native block map, never auto/balanced/offload inference."""
    if type(layers)is not int or type(first_layers)is not int or not 1<=first_layers<layers:
        raise ValueError('explicit nonempty contiguous decoder partitions required')
    mapping={'model.visual':'cuda:0','model.language_model.embed_tokens':'cuda:0',
        'model.language_model.rotary_emb':'cuda:0'}
    mapping.update({'model.language_model.layers.'+str(n):'cuda:0'if n<first_layers else'cuda:1'for n in range(layers)})
    mapping.update({'model.language_model.norm':'cuda:1','lm_head':'cuda:1'})
    return dict(kind='explicit_two_cuda',device_map=mapping,trainable_device='cuda:1',first_layers=first_layers,
        owned_devices=[{'device':'cuda:0'},{'device':'cuda:1'}])


def validate_placement(placement,config):
    if not isinstance(placement,dict):raise ValueError('explicit product-owned placement required')
    if placement.get('kind')=='single':
        if placement!=single_placement(placement.get('device')):raise ValueError('exact single-device map required')
    elif placement.get('kind')=='explicit_two_cuda':
        layers=config['text_config']['num_hidden_layers']
        if placement!=two_device_placement(layers=layers,first_layers=placement.get('first_layers')):
            raise ValueError('exact complete native decoder/embed/vision/norm/head boundary map required')
    else:raise ValueError('no auto/balanced/quantized/CPU-offload placement')
    return placement


def rate_overrides(options):
    values={key:options[key]for key in('lr','d_lr_mult','prior_lr_mult')if key in options}
    for key,value in values.items():
        if type(value)not in(int,float)or not math.isfinite(value)or value<=0:raise ValueError('finite positive non-Boolean '+key)
    return values


def validate_profile(options):
    if(options.get('source_profile'),options.get('initialization_policy'),options.get('pg_version'),
        options.get('pg_source_profile'),options.get('penalty_input_domain'))!=(SOURCE_PROFILE,INITIALIZATION_POLICY,'0.9.5','release-0.9.5','continuous'):
        raise ValueError('explicit new Arete095 fresh-init/continuous native profile required')
    if 'ema_decay'not in options or options['ema_decay']is not None or options.get('ema')is not None:
        raise ValueError('explicit ema_decay=None required; all weight EMA refused')
    rate_overrides(options)


def validate_native_options(options,*,training=False):
    allowed={'model_dir','placement'}|set(DEFAULTS)
    if training:allowed|={'training_file','max_tokens','enable_thinking','steps','batch','seed','ema_decay','source_profile',
        'initialization_policy','pg_version','pg_source_profile','penalty_input_domain','lr','d_lr_mult','prior_lr_mult'}
    if set(options)-allowed:raise ValueError('unknown Arete095 option; use literal common rates and explicit placement')
    if training:validate_profile(options)
    arch=architecture(options)
    if (arch['rank'],arch['particles'],arch['particle_dim'])!=(8,128,4)or any(type(arch[key])is not int for key in('rank','particles','particle_dim')):
        raise ValueError('preserve actual shared128x4/rank8 architecture')
    if type(arch['response_width'])is not int or arch['response_width']<1 or arch['base_precision']not in('float32','bfloat16')or type(arch['checkpointing'])is not bool:
        raise ValueError('full positive native response width and unquantized FP32/BF16 required')
    root=Path(options.get('model_dir',''))
    if not isinstance(options.get('model_dir'),str)or not root.is_dir():raise ValueError('explicit local model_dir required')
    config=json.loads((root/'config.json').read_text())
    if config.get('model_type')!='qwen3_5'or config['text_config']['hidden_size']!=arch['response_width']:
        raise ValueError('response width must equal the complete native Qwen3.5 hidden width')
    validate_placement(options.get('placement'),config)
    if training:
        for key,default,minimum,maximum in(('steps',4,1,2**31-1),('batch',2,2,512),('seed',7,0,2**63-100),('max_tokens',4096,1,32768)):
            value=options.get(key,default)
            if type(value)is not int or not minimum<=value<=maximum:raise ValueError('invalid bounded '+key)
        if type(options.get('enable_thinking',False))is not bool:raise ValueError('explicit Boolean thinking choice required')
        if not isinstance(options.get('training_file'),str)or not Path(options['training_file']).is_file():raise ValueError('matched local training_file required')


def validate_installed_core():
    from importlib import metadata
    lock=json.loads(Path(__file__).with_name('current_core.json').read_text());dist=metadata.distribution('particle-sliders-core')
    if dist.version!=lock['version']or lock['commit']!=CORE_REVISION:raise ValueError('exact current core0.14.1 source cohort required')
    for name,expected in lock['checked_api_files'].items():
        if file_hash(dist.locate_file('particle_sliders/'+name))!=expected:raise ValueError('current core Source differs:'+name)
