# CALIBRI v186 — balanced opposing-response source QAT (2026-11-05)

**Measured technical PASS**, **original-FP32-like answer quality FAIL**.
The source really changes bits while staying at the original
0.667839 bpp and native CPU speed. Balancing opposite answers
partially prevents universal `Yes` collapse on one narrow
behavioral domain; arithmetic still collapses to repeated `5`.

[Actual completed GitHub Actions #42085087419](https://github.com/Kamara229/JessicaCore/actions/runs/42085087419)
conclusion SUCCESS after the first workflow attempt
[#42084808632](https://github.com/Kamara229/JessicaCore/actions/runs/42084808632)
FAILED to handle an arithmetic pair with the same first BPE token.
The issue was corrected by constructing different-leading-token
arithmetic contrasts **before any evaluation**, with TRAIN/VALID/DEV
splits preserved.
[Full source + original unfiltered answers artifact #13443107323](https://github.com/Kamara229/JessicaCore/actions/runs/42085087419/artifacts/13443107323)
contains physical archived code, 96-step TRAIN audit,
all 12 held-out VALID pairs, all 24 historical DEV generated
answers per arm, native CPU timing and source checksum.

## Physically measured 5-bit code source

Original pretrained SmolLM2-135M: **134,515,008 unique parameters**,
original source CB59 11,229,303 complete bytes. v186 uses exact
existing packed format K32/V8 (one 5-bit code per 8 trained
source scalars) and physically unchanged codebook/norm/scale
allocation. **No 2D FP32 weights resident in native CPU runtime**.

- Final archive **11,229,303 bytes = 0.66783941313 bits/original
  unique parameter**; physically under <=0.700.
- SHA256
  `933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a`.
- **30,208** original 5-bit group indices changed;
  **53,906** FP16 codebook cells changed, verified against
  immutable baseline. No unpaid sidecar or updated 1D norms.
- 96 locked whole-network gradient steps, **24 balanced
  opposite-answer public TRAIN pairs**, 12 entirely disjoint
  public VALID pairs, original 24 historical DEV questions
  not used in source optimization.
- Saved physical native C execution versus same TRAIN
  surrogate full-model logits relative L2 **5.7214e-7**,
  same greedy first token; establishes numerical source
  equivalence, **not QA correctness**.
- Median native CPU speed **99.59% of original CB59**
  across 8 matched 32-token CPU benchmarks, including
  prefill; >=90% speed criterion *for these runs*.

## Genuine held-out VALID contrast, before historical DEV

| First-token criterion, 12 paired VALID questions | FP32 | CB59 | CB186 |
|---|---:|---:|---:|
| Both opposite answers correct | 0/12 | 0/12 | **3/12** |
| Actually changes first token for opposite pair | 0/12 | 0/12 | **3/12** |
| Individual expected first tokens | 9/24 | 0/24 | **12/24** |
| Number of unique first token IDs | 4 | 2 | **4** |

All **three** fully correct opposing pairs in CB186 are from
fictional Yes/No possession/binding domain; **0/3** numeric
comparison, **0/3** arithmetic and **0/3** color binding
pairs both correct. This is the first verified *nonconstant*
opposite-token response for one held-out behavioral
subtype in this late series. However this binary-target
metric is sensitive to exact token formatting. The FP32
0/12 score does **not** establish that the original
pretrained model understands none of these questions.

Actual archived VALID generation examples:
- Fictional **silver book yes/no**: same entity carrying
  silver versus gold -> **`Yes` and `No`** for CB186
  (both correct).
- Numeric `Is 1103 greater than 1100?` versus its
  reversal -> CB186 **`No` and `No`** (first wrong).
- Arithmetic `723 + 5` versus `723 + 205` ->
  CB186 **`5555555555555555`** on BOTH.
- Color `red` versus `blue` folder swap ->
  CB186 **`red` and `red`**, not both correct.

### Historical 24-case public DEVELOPMENT QA (still FAIL)

| Actual 32-token generated-answer measure | FP32 | CB59 | CB186 |
|---|---:|---:|---:|
| First-token FP32 matches on 20 fixed DEV | 20/20 | 0/20 | **1/20** |
| Distinct first tokens on same 20 | 10 | 3 | **4** |
| Machine-confirmed correct generated answers /24 | 6/24 | 0/24 | **2/24** |
| Machine parseable answers /24 | 15/24 | 0/24 | **12/24** |
| Responses that hit cap without EOS /24 | 4/24 | 24/24 | **12/24** |
| Mean emitted-token distinct fraction | .8948 | .0508 | **.5156** |

Original FP32 also has **9 ungraded** historical answers;
CB186 has **12 ungraded**. Do not falsely call all ungraded
answers incorrect.

The two machine-confirmed CB186 responses were:
- Historical fictional notebook binding #16 -> **`red`**.
- Historical negative numeric comparison #83 ->
  **`No`**.

On all 8 historical arithmetic prompts CB186 repeated
`555555555555555555...`, including arithmetic questions
whose correct answers were not 5. Other DEV outputs are
often generic `Yes`/`No`/`red`.
There is NO general recovered reasoning, factual answering
or original-FP32 answer parity. These 24 questions are
historical development, not independent private TEST.

## Scientific interpretation and next gate

Balanced paired supervised TRAIN recovered *one narrow*
opposite-answer distinction and partially reversed global
`Yes` dominance, but **caused other constant-output modes
and not robust generalization across task domains**.
One must not call 3/12 VALID pair successes a complete
language model success.

A useful next controlled experiment is an original-FP32
teacher *capability audit* on a broader balanced independent
public question benchmark before selecting a bigger model,
then simultaneously investigate a true multidimensional
source geometry (v187 pilot) or a larger, better trained
compact native representation. Current SmolLM2-135M
FP32 is itself weak on the historical 24 prompts. Avoid
fitting any code on private v89 TEST or reusing public
DEV answers for source updates. No NLL/KL/PPL acceptance;
actual meaningful responses and bytes/speed remain decisive.

Research-only draft PR #324, no merges/deployments.
No main, LIVE Jessica, Render, Supabase, APK or Green
Tunnel touched. Jessica is literal relay only and
ran NO model tests.
