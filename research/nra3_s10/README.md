# NRA2 3.0 S10 — authentic original CALIBRI 43M post-GT layer8 for 72 NEW Russian TRAIN sentences

**RESEARCH ONLY / DRAFT / DO NOT MERGE. No LIVE or application settings.**

The S8 two router versions both collapsed to global selection, regardless of
in-sample loss reductions. S9 registered exactly 72 **new and templated TRAIN**
Russian sentences with complete UD and semantic annotations; this is NOT
an unseen benchmark or a natural-language proficiency test. Their exact full
canonical source SHA is
`b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d`,
pinned by GitHub Actions S9 #43731968521.

## What S10 does

S10 reuses the previously successful **S5 ORIGINAL real 43M CPU S194**
workflow verbatim for artifact provenance and original 616 J7 verification:
456 original protected responses and 128 new protected responses (with full
EOS), plus the original separate 32 TRAIN probes. The frozen 43M weights,
S93 Green Tunnel and all-layer bridge remain unchanged. Only **after**
the original 616 J7 complete output and digests are collected, S10 runs
72 new prefill calls, registered **inside** the real S93 / existing GT
read-only forward scope. A `register_forward_hook` on layer #8 (zero-based
index 7), installed **after** the existing GT/S103 hooks, returns
`None` and copies the original output to FP16 CPU. No original Calibri
logits or returned PyTorch tensors are changed.

Each exact prompt is run through the **actual original Calibri fast tokenizer**
using its chat-template; expected CPU prefill IDs and exact UD word offsets
must match, with no token crossing two annotated words. The original 72
TRAIN-only text prompts go into Calibri. The optimizer-only UD/gold role
labels are **never** passed into its forward.

The physical artifact contains only `[72,T,320]` FP16 hidden, real token
IDs, token masks/lengths, and an JSON report of external SHA and wordpiece
positions. Gold role/source information never appears in the source
hidden tensor. All 72 rows must be physically saved with SHA256, and
reloaded exactly; a missing row or altered original 616 generated token
checksum fails closed.

## What S10 cannot yet demonstrate

No new NRA3 training, no trained selector, no candidate generation, no true
unseen Russian generalization, no new full answer + EOS quality, no
whole-parent+GT+NRA all-in <=0.7 bpp / CPU>=90% certification. The existing
S8 global-collapse candidate remains **rejected**. Its output gate stays
`0`; `candidate_accepted=false`.

The five previously consumed independent EVAL128 and reserved sixth S3CI
EVAL128 are NOT loaded or used. No ordinary Jessica chat tests, main
branch, LIVE, Render, Supabase, APK or original Calibri/GT writes.

### Next research-only phase

After S10 success and physical SHA signature, train a **small physically
billed selector head** on the 72 new original43M hidden vectors and
contrastive weak targets, while preserving original 43 5-bit code streams
and avoiding the S8 collapsed hard selections. New independent evaluation
must be preregistered *before model choice*, not generated from familiar
lexical families or TRAIN templates.
