# CALIBRI v0.8 — real language model held-out quality

Separate research branch derived from v0.7, **not merged and never deployed to Jessica**.

Train/calibration: 512 WikiText-2 TRAIN tokens fed through the exact trained
Pythia-14M model. Held-out quality: 1024 WikiText-2 VALIDATION tokens.
Model and dataset revisions, token-hash digests are saved at runtime.

All 12 MLP projections are ternarized at 50%, 75%, and 88.5% **imposed**
zero rate. For each: magnitude-only, activation-column-energy weighted,
and activation-weighted + 2-level least squares on calibration activations.

Reports WikiText validation non-overlap-block perplexity (FP baseline,
all 9 quantized combinations), per-MLP CL8-I sizes including scales and
indexes, estimated mixed-checkpoint storage versus FP16 original.

Severe limitations: Pythia-14M (dense) is NOT Switch-MoE; quantization is
not QMoE GPTQ. Perplexity is evaluated by dense reconstructed weights,
**not** compressed execution. 1024 validation tokens is a tiny diagnostic
sample, not a publishable perplexity benchmark. Total checkpoint sizes
are ESTIMATES using FP16 for other parameters. Training and validation
splits are kept separate. No GPU performance is asserted.

Workflow `.github/workflows/calibri-quality-v08.yml` is isolated to PR
and explicit workflow_dispatch; never production Render/Supabase/skills.
