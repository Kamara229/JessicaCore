# CALIBRI 3.0 research

## Mandatory engineering workflow (how we work on CALIBRI)

1. Develop functionality in small focused source files. Compose related modules using an architectural orchestration file, and compose such files at higher levels as needed. This is a rule for organizing our development work and source code, NOT a principle of CALIBRI's neural architecture, inference, learning algorithm, layer connectivity or model execution. Do not introduce neural modules, runtime wrappers, extra calls, or architectural constraints merely to mirror the source-code hierarchy. Use the simplest efficient runtime implementation.
2. Improve the model sequentially from one validated version to the next. Each successor pins the prior physical checkpoint, records changes, and verifies preservation of learned capabilities. Failed candidates are retained for research but do not replace the validated lineage.

## Model constraints

12 neural layers fixed. No reset, no production activation.

V001 uses the last confirmed G43 Balanced physical checkpoint (G44 still running when selected). Computational experiment only, no model optimization or claims of improved generation. Preserve GT, NRA3, G41 balance, 2-token head and all G43 weights. If G44 subsequently passes its gates, evaluate it as an explicit successor/migration, not silently overwrite this parent.

Current V001 source-code organization (development only, not neural topology):
- weights/compact.py: direct codebook operator, bounded gather, autograd.
- weights/profile.py: counted/timed legacy reconstruction and clear; separate benchmark.
- weights/architecture.py: assemble and test operators.
- lineage.py: exact checkpoint restoration and identity.
- architecture.py: whole-model orchestration and report.
- run.py: CLI only.

Next: explicit freeze and versioned cache; actual batch ABI; one-pass causal supervision; compact operator optimization (Metal/MLX only after parity); bank attribution and GT/NRA3 integration; adaptive code indices and replay. Each change requires a matched baseline and preserved physical checkpoint. No FP32 teacher. Keep known physical payload <=0.7 bpp, verify full runtime bytes separately, CPU speed >=85–90% baseline, and historical J7 616 independently. No architecture promotion until all global gates.
