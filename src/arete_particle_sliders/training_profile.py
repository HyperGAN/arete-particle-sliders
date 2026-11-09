"""Allocation-free discovery for the shared particle-slider command."""
TRAINING_PROFILE = dict(
    schema='particle-slider-profile-v1', name='arete-cpu-fixture',
    product='arete-particle-sliders', config_kind='session', status='fixture',
    interface='response', backend='arete-particle-sliders', runtime={},
    capabilities=dict(resume=True, periodic_checkpoint=True, check=True),
    default_checkpoint_every=2,
    qualification='Random CPU Qwen3.5 architecture fixture; pretrained/GPU unqualified')
