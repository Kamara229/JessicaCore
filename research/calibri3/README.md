# CALIBRI 3.0 research

Sequential lineage, 12 layers fixed. Small leaf modules compose through local architecture.py files, then root architecture.py; no monolithic experiment implementation. Every successor pins prior serialized model and records changes. No reset, no production activation.

V001 uses the last confirmed G43 Balanced physical checkpoint (G44 still running when selected). Computational experiment only, no model optimization or claims of improved generation. Preserve GT, NRA3, G41 balance, 2-token head and all G43 weights. If G44 subsequently passes its gates, evaluate it as an explicit successor/migration, not silently overwrite this parent.

Pyramid:
- weights/compact.py: direct codebook operator, bounded gather, autograd.
- weights/profile.py: counted/timed legacy reconstruction and clear; separate benchmark.
- weights/architecture.py: assemble and test operators.
- lineage.py: exact checkpoint restoration and identity.
- architecture.py: whole-model orchestration and report.
- run.py: CLI only.

Next: explicit freeze and versioned cache; actual batch ABI; one-pass causal supervision; compact operator optimization (Metal/MLX only after parity); bank attribution and GT/NRA3 integration; adaptive code indices and replay. Each change requires a matched baseline and preserved physical checkpoint. No FP32 teacher. Keep known physical payload <=0.7 bpp, verify full runtime bytes separately, CPU speed >=85–90% baseline, and historical J7 616 independently. No architecture promotion until all global gates.
