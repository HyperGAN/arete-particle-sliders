"""Native hybrid attention only; one bank, original shared routed projections."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from particle_sliders.adapter_activation import push_activation,restore_activation
from particle_sliders.adapter_attachment import atomic_attachment
from particle_sliders.routed_projection import TinyParticleProjection
from .targets import native_attention_targets


class AreteSlider(nn.Module):
    """Keep all trainables on one explicit device, even for a split frozen base.

    The unchanged shared projection transfers the full input to its own FP32
    branch and the full delta back to the original host device. Cross-device
    copies stay differentiable; no hidden coordinates or bank copies are cut.
    """
    @atomic_attachment('model')
    def __init__(self,model,*,trainable_device):
        super().__init__()
        if any(hasattr(m,'_arete095_owner')for m in model.modules()):raise ValueError('Arete already attached')
        plan=native_attention_targets(model)
        self._modes=[(m,m.training)for m in model.modules()]
        self._grad=[(p,p.requires_grad)for p in model.parameters()]
        model.eval().requires_grad_(False)
        # Historical Arete law is unit Normal, deliberately not Sophon's std0.1.
        self.particles=nn.Parameter(torch.randn(128,4,device=trainable_device,dtype=torch.float32))
        self.projections=nn.ModuleList();self._hosts=[];self.target_names=tuple(plan)
        for name,host in plan.items():
            projection=TinyParticleProjection(name,host,self.particles,8,8.).float().to(trainable_device)
            object.__setattr__(host,'_arete095_owner',self)
            self.projections.append(projection);self._hosts.append((host,projection,host.forward))
        self.closed=False

    @contextmanager
    def scaled(self,strength):
        if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1 or self.closed:
            raise ValueError('open unipolar native slider and strength0..1 required')
        push_activation(self,self.projections,('multiplier',))
        try:
            for projection in self.projections:projection.multiplier=float(strength)
            yield self
        finally:restore_activation(self)

    def close(self):
        if self.closed:return
        if getattr(self,'_activation_stack',None):raise RuntimeError('cannot detach active slider')
        if any(getattr(host,'_arete095_owner',None)is not self or host.forward!=attached for host,_,attached in self._hosts):
            raise RuntimeError('Arete forward ownership changed; do not remove foreign hooks')
        for host,projection,_ in reversed(self._hosts):host.forward=projection.org_forward;del host._arete095_owner
        for p,flag in self._grad:p.requires_grad_(flag)
        for m,flag in self._modes:m.training=flag
        self.closed=True
        # The owner has detached successfully; retain adapter Parameters but no
        # bound native forward/module/Parameter references to the huge base.
        for projection in self.projections:projection.org_forward=None
        self._hosts.clear();self._grad.clear();self._modes.clear()
