# Source notices

The original Arete host helper is copied byte-for-byte from
HyperGAN/arete-particle-sliders@37e6a701 (MIT), with its extraction identity in
`src/arete_native95/host-source.json`; the original MIT license is included.

`initialization.py` derives from HyperGAN/sophon-particle-sliders local source
175edeeade49f1e4044dc76f6a7f156cb1928482 (Apache-2.0), adapted for the new Arete
namespace, profile, and original Arete unit-Normal bank law. Its Apache-2.0
license is included at `LICENSES/Sophon-Apache-2.0.txt`. No prior Sophon
qualification is transferred. Shared core routed/lifecycle/initialization APIs
are imported, not vendored or rewritten.

The model is NousResearch/Arete-27B (Apache-2.0). No model tensors, data or
pretrained weights are included or redistributed by this source project.
