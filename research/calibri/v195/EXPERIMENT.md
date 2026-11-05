# CALIBRI v195 — 360M physical native C vs exact decoded source, then unpaid FP32 family causal controls

## Verified motivation
The actual 360M complete 0.648907 bpp physical archive is **real**
(29,348,539 bytes, SHA `69e2e0f...`). Its native C
model actually emits 16/16 illegible repetitive strings, 0 EOS
and 0 grader-confirmed correct; original FP32 got 8/16 on
the exact same questions. This is a catastrophic **quality FAIL**.

We do NOT know if destruction is caused by numerical differences
in C compressed lookup-GEMV vs the source decoder or inherently
destructive neural quantization, or where it is concentrated.

## Diagnostic step A — native numerical integrity
1. Reuse the EXACT previous 360M archive, not recompress
   or alter a weight. Verify physical length, SHA, CRC,
   tied aliases, all original 361,821,120 unique weights.
2. At precommitted real tensor names (attention q_proj,
   mid MLP down_proj, final MLP gate_proj) compute
   C-native packed GEMV and a separately constructed
   **DENSE DIAGNOSTIC** oracle from the same CB59
   centroids/codes/scales, with fixed realistic CPU
   activation vectors. Require a small absolute/relative
   numerical error budget; dump both errors.
3. For a sampled set of actual embedding rows, compare
   C on-demand embedding decoder with another exact
   decode of the same 5bit packed indices. Do not
   decompress the entire native model in production.
4. This verifies numerical implementation, NOT
   original-FP32 fidelity or meaning. If C diverges,
   fix the decoder first; no conclusions on quantizer.

## Diagnostic step B — original FP32 family causal rescue
Use a **fresh 360M original pretrained FP32 checkpoint**
**only** in this experimental diagnostics job, never claim
it is physically compressed. A separate compact shell
uses exact v194 paid CB59 binary in all non-substituted
2D modules and the C-native decode. Swap in a subset
of reference original FP32 modules WITHOUT training:
- original shared input embedding and output lm_head only;
- all attention projections only;
- all MLP projections only;
- all attention & MLP but **keep embedding/head packed**;
- original shared embedding/head and attention, MLP packed;
- original shared embedding/head and MLP, attention packed.

Compare actual 8 greedy generated texts per intervention
(2 each arithmetic, reversed greater-than Yes/No,
reversed owner/color binding and factual) and original
v193 FP32 golds. Preserve ALL raw EOS/cap states.
Use only previously SHA-locked v192 grader, no
posthoc edits. Keep source/rate manifest and actual
FP32-injected module/parameter counts for each arm;
**ANY ARM WITH ORIGINAL FP32 MODULES IS AN UNPAID
DENSE DIAGNOSTIC, NEVER A VALID <=0.7 BPP MODEL.**
A score improvement in an arm only locates a damaged
component family and guides a future **physically paid**
repair operator.

## Interpretation / research separation
No private v89 TEST, no hidden feedback label fitting,
no new original QA source selection, no NLL/KL/PPL gate.
Do not use results to claim source generalization.
CI uses research-only unmerged PR; main, production
Jessica/Render/Supabase/APK and Green Tunnel unchanged.
Jessica relay must never execute these tests.
