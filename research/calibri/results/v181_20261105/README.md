# CALIBRI v181: completed real pretrained high-rate recovery controls (5 Nov 2026)

**Scientific verdict: FAILED answer recovery at BOTH 2.54 and 5.04 bpp.**
These two above-budget controls are diagnostic only, never candidates
for the final <=0.700 bits/original parameter requirement.

## Verified original-source results

- Source model: actual pretrained SmolLM2-135M, 134,515,008 unique original
  parameters; immutable original CB59 SHA256
  `50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c`.
- Completed successful GitHub Actions [run 42063234221](https://github.com/Kamara229/JessicaCore/actions/runs/42063234221);
  native unit tests passed, all 211 actual matrices were saved and
  tested, 24 actual developmental prompts per arm (4 arms).
- [Raw artifact 13434816619](https://github.com/Kamara229/JessicaCore/actions/runs/42063234221/artifacts/13434816619)
  includes complete physical V1/V2 source archives, original source
  provenance, native C numerical checks and all generated answer texts,
  token IDs, prompt SHA and grades. Treat artifact retention as temporary.

| Measured | FP32 | Original CB59 | ORACLE_V2 | ORACLE_V1 |
|---|---:|---:|---:|---:|
| Physical archive bytes | FP32 source | 11,229,303 | **42,668,036** | **84,679,302** |
| Bits/original parameter | 32 | 0.66784 | **2.5375926** | **5.0361252** |
| First-token matches FP32, 20 DEV prompts | 20 | 0 | **0** | **0** |
| Unique first tokens in 20 DEV prompts | 10 | 3 | **4** | **14** |
| Machine-confirmed correct answers, 24 actual DEV prompts | **6/24** | 0/24 | **0/24** | **0/24** |
| Responses reaching 32-token cap without EOS | **4/24** | 24/24 | **24/24** | **24/24** |
| Mean generated-token distinctness | .8948 | .0508 | **.3190** | **.6745** |

Original FP32 6/24 is a conservative frozen machine-graded tally,
not exhaustive semantic accuracy. 24 actual questions were tested:
**do not claim 0/96**. The 96-question development suite and independent
private v89 TEST were not run for v181.

Sealed candidate source SHA256:
- ORACLE_V2 `029258a6396662d6bbe241c715cd1045fe3d6c68507302e3f56ed0a95d70765d`
- ORACLE_V1 `1832c303f2655ee52d335ea8a3cbdda76756abcacc598573f4462cf7703451d8`

## Source-to-FP32 relative weight errors

Unweighted mean *per-matrix* relative source weight RMSE, recomputed
from the saved `sealed_ORACLE_V1.json` and `sealed_ORACLE_V2.json`
with 211 matrix entries each. This is a diagnostic, **NOT** an
acceptance metric:

| Source matrix family | V2 (5b/2 weights) | V1 (5b/1 weight) |
|---|---:|---:|
| Shared embedding/output | .26925 | **.12901** |
| 120 attention matrices | .26498 | **.10360** |
| 60 MLP gate/up | .25841 | **.09810** |
| 30 MLP down | .26159 | **.09733** |

V1 considerably improves source fidelity and first-token diversity
but actual generated text is still incoherent. Example on
`Calculate 7 plus 3.`: FP32 says `7 + 3 = 10` and finishes,
while V1 starts unrelated programming words and never emits EOS.
The original CB59 repeats `speci`; V2 repeats `ong`.
None of the three compressed sources recovered task-relevant answers.

The compiled C kernel directly executes the fully physical packed
V1/V2 sources without dense original trained 2D weight matrices.
Numerical native-vs-exact-decoded-source unit tests passed. That
does not establish that the *decoded source weights* preserve correct
model behavior. Do not equate technically successful CI with quality.

## Crucial scientific inference and next go/no-go

**It is not enough to increase V8 index precision to V1 while keeping
a global K32 dictionary per matrix and nonlearned row scaling.**
Even at 5.04 bpp, this specific representation fails real QA,
so it is misleading to treat its failure as a general proof that
5bpp cannot retain pretrained LLM behavior.

Next controlled experiment must separate source corruption from
native runtime/model binding: compare SAME model FP32, uncompressed
FP16 tensors, and V1 decoded FP32 matrices using the same exact
PyTorch architecture, then verify native V1 matches decoded V1
per-layer activations and next-token logits. Locate FIRST divergence
by layer on frozen PUBLIC development prompts. Only afterward choose
new learnable quantization or structured source formats to pursue
<=0.7 bpp. Without this control, further low-bit representation
changes have too little diagnostic value.

Never use KL/NLL/PPL as final acceptance gates; actual coherent
answers relative to original FP32, <=0.7 total physical bpp and
>=90% measured native CPU speed remain essential. No main/LIVE
Jessica, Render, Supabase, APK, Green Tunnel or private TEST changes.
Jessica remains literal chat relay only, NEVER a test runner.
