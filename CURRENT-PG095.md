# Separate native Arete PG0.9.5 cohort

The original root PG0.9.4 code, locks, fixtures and evidence remain unchanged;
this documentation adds the later native cohort and measured scope. Install the separate [native-candidate project](native-candidate/README.md)
for namespace `arete_native95`: full5120 continuous responses, the original
Qwen3.5 conditional model, explicit two-device placement, fresh-only scoped
initialization, RAW/noEMA, exact FULL continuation and native finished text.

**Nine distinct native tiny CPU cases passed:** eight retained at d9f9a4d7 plus
one corrected finished-generation case at5dfd5459. Old passing cases were not
rerun. [Portable actual evidence](native-candidate/evidence/cpu-nine-distinct-qualification.json)
records both closed outcomes, original artifact names/bytes/SHA256, exact
versions, cases and limits. This is tiny random real-architecture software
qualification, not full27B/pretrained/actual-twoGPU/quality/LR qualification.

Retained pretrained execution now includes two RAW32 arms and native finished
texts. The [completed provisional comparison](docs/finished-output-learning-rates.md)
selects saved G0.012 (+1.0) over G0.001 (0.0) on two equal cases; both FULL32
catastrophic guards pass, while `trait_qualified:false` remains. This does not
make the nine tiny CPU cases a pretrained proof, change a package default,
release weights or qualify a global effect. Preserve every old source/score and
the original initialized receipt on FULL restore/RAW inference.
