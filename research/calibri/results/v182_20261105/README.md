# CALIBRI v182 — completed first-layer divergence study

**Diagnostic CI PASS. Answer-quality recovery remains FAIL.**
Full real pretrained SmolLM2-135M on frozen public development prompts
0, 16, 64 and 80, four different question types.

- [GitHub Actions run #42069314531](https://github.com/Kamara229/JessicaCore/actions/runs/42069314531): SUCCESS, four focused unit tests passed.
- [Complete raw metrics #13434539776](https://github.com/Kamara229/JessicaCore/actions/runs/42069314531/artifacts/13434539776): 30 layers, first-two-layer projection probes, source check, actual first-token IDs.
- Original physically stored V1 bytes **84,679,302** (5.036125 bpp).
- V1 SHA256 **1832c303f2655ee52d335ea8a3cbdda76756abcacc598573f4462cf7703451d8**.

| Compared against original FP32 | Greedy first-token matches (4 public DEV) | Mean last-token logits relative L2 | First >5% hidden-state divergence |
| --- | ---: | ---: | --- |
| FP16-rounded same original pretrained model | 4/4 | 0.0000007615 | None |
| V1 fully decoded into PyTorch FP32 matrices | 0/4 | 0.672863 | Input embedding |
| V1 genuinely physically packed native CPU C | 0/4 | 0.672861 | Input embedding |
| V1 tied embedding/head; all remaining original FP32 | 3/4 | 0.278629 | Input embedding |
| Original FP32 tied embedding/head; remaining 210 V1 matrices | 0/4 | 0.394559 | First transformer block |

Native V1 versus **the very same decoded source** in conventional
PyTorch yields all four *identical first-token IDs* and max
relative L2 difference **0.000029278** over all 33 main
checkpoints. Hence C-native weight execution is not materially to
blame for the model's nonsense generations.

With FP32 input/head preserved, first decoder-block hidden
activations already diverge ~0.398–0.413 relative L2 on the four
prompts. Inside first block, attention O-projection differs
~0.446–0.450, MLP gate projection ~0.953–1.016, MLP up
~0.743–0.758. The V1 source changes meaningfully before later
layers; errors then accumulate.

Source weight relative L2: V1 tied embedding/head **0.129006**,
first MLP down projection **0.080152**. These matrix errors and
four first-token comparisons are **diagnostics, not correct-answer
measures**; v181 real QA still 0/24 for complete V1. The 4
development prompts are not an independent blind test. Activation
hooks sampled final prompt position only.

Next experimental gate: identify most harmful first-block body
projection with single-operator full-FP32 counterfactuals, label
these diagnostic copies **unpaid and nondeployable**, then design
a physically costed compact learned source representation. Do
not revive discarded KL/NLL/PPL as CALIBRI answer gates.
The final target remains <=0.700 *full physical* bpp,
coherent original-FP32-like answers and native CPU >=90% CB59.

All changes are in research-only draft PR #320. No main,
LIVE Jessica, Render, Supabase, APK, Green Tunnel, private TEST
or Jessica chat-based model tests were touched.
