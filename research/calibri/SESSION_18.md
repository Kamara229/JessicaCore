# CALIBRI — session 18: previously unseen corpus, frozen candidates

## Why another test?

v17 GitHub Actions, real pretrained Pythia-14m one 128x512 MLP:
- **Six geometries** passed the predeclared WikiText-2 validation
  +1% PPL gate and ≤49,563-byte storage ceiling.
- Smallest passing: m2=16,n2=32,rank2,INT5 residual: 43,832 bytes,
  dev PPL 173.478 vs original 173.381 and INT6 173.270.
  However the previously observed WikiText-2 test PPL was **255.015**
  vs original 247.540 and INT6 248.685. This FAILED to generalize well.
- Best WikiText development quality: m2=8,n2=32,rank8+INT5 residual
  49,464 bytes, dev PPL 172.034 and exploratory test 247.460.
  Note this is **38 bytes larger** than INT6, and slight PPL changes
  on 1,016 test tokens are noisy, not proof of better general quality.
- A different preapproved geometry (m2=8,n2=16,rank4,INT5 residual)
  uses 46,392 bytes with dev PPL 172.817: a possible size/quality
  Pareto candidate vs simple INT6. Its old WikiText test was NOT
  inspected before this preregistered cross-corpus evaluation.

Therefore strong independent evidence is required before claiming even
a small win. Critically, do not modify the methods on new data.

## Exact fixed manifest BEFORE the new corpus is inspected

Benchmark on a previously **unused** dataset
\`roneneldan/TinyStories\` validation split, pinned HF revision.
Only the following frozen options from v17:
- INT5 (control), INT6 (control)
- Geo-small: m2=16,n2=32,rank2,residual INT5; 43,832 B
- Geo-balanced: m2=8,n2=16,rank4,residual INT5; 46,392 B
- Geo-accuracy: m2=8,n2=32,rank8,residual INT5; 49,464 B

No candidate selection or training on this dataset. Evaluate 8,192
tokens using a 128-token context and report reference LM perplexity,
absolute physical candidate size (including all metadata), relative
perplexity and independent distribution shift behavior.

## Limitations and next steps

TinyStories is synthetic children's-story text, qualitatively different
from WikiText Wikipedia. Positive results only show transfer under this
domain, not general quality or a patented invention. One dense MLP
projection, no MoE, no shared expert graph and no ARM/Android backend.
Residual still executes as dequantized dense matrix in the pilot.
Native direct calculation only applies to TT core.

Any future human-facing claim needs several diverse held-out suites,
real MoE expert routing, fused kernels, full-model accounting, and
comparison against GPTQ/AWQ/current tensor methods.

**Safety:** draft research branch only; no main merge/Render/Supabase/
mobile updates, no personal user data or model upload.

## Completed independently sourced new-corpus results (Actions 41336227016)

Fresh TinyStories validation, 8,192 tokens (8,128 scored), 141 examples,
context 128. The methods were *fixed before* this corpus was inspected.

| Model variant | Physical bytes | TinyStories PPL |
|---|---:|---:|
| original | 131,072 FP16 weight estimate | 37.04143 |
| INT5 | 41,234 | 37.83302 |
| INT6 | 49,426 | 37.20436 |
| Geo-small (r2, INT5) | **43,832** | **37.04550** |
| Geo-balanced (r4, INT5) | 46,392 | 37.24036 |
| Geo-accuracy (r8, INT5) | 49,464 | 37.56816 |

Geo-small is 11.32% smaller than INT6 and performed better on this
specific domain, nearly matching original PPL. However, the *same*
configuration scored 255.015 on the earlier WikiText-2 test versus
248.685 for INT6. This **cross-domain contradiction is central**:
there is no demonstrated universally quality-preserving replacement.
Mixed 5-bit residual is still restored to a dense buffer at runtime,
so storage savings do not establish RAM/speed savings. The sample is
small and lacks statistical uncertainty estimates; repeat on diverse
fresh domains and a true MoE before deploying.
