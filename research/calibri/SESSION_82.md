# CALIBRI v82 — independent broader QA and generation generalization of v81

**Preregistered 3 November 2026, before ANY v82 TEST evaluation.** Research-only GitHub draft PR. No `main`, live Jessica, Render, Supabase, APK, training or inference of user-facing model are changed.

## V81 verified, what v82 is and is not
The successful real original pretrained SmolLM2-135M-Instruct, 134,515,008 unique trained original parameters CI **41555699743** achieved **one** encouraging but very small heldout answer result: fixed parent v76 CB76 BoolQ **9/32** → new v81 **17/32**, SciQ **6/27** → **7/27**, while genuine pretrained FP32 teacher scored **25/27** SciQ and **18/32** BoolQ. New v81 corrected model had worse heldout full-vocab KL/NLL than prior v76 on both WikiText and AG News; the WikiText KL regression +0.3635 exceeded its reused TRAIN-CAL +0.25 guarantee. Greedy token generation remained extremely repetitive (**13.9% unique token IDs** vs pretrained teacher 86.1%). This does **not** establish restored assistant competence.

The physical C-native 5bit base CB59 source remained **11,229,303 B / 0.667839413 bpw**, preserving all original pretrained trained 2D source and aliases/checksums. One fully runnable paid CB76 source with FP16 `U[49152,4]`, `V[576,4]` and bias + file header/integrity was exactly **11,725,487 B / 0.697348923 bpw**, including **all** source bytes. The paid correction uses CPU Torch; only original base operators are direct native C. No fully-native, quality, speed or larger-weight model claims.

**v82 is a falsification/replication stage**, NOT a new trained candidate or codec innovation. Its purpose is to decide whether the v81 raw heldout BoolQ result is likely to generalize to *different*, somewhat larger unseen question sets without undoing language or worsening token repetition. We must not tune factor weights, prompts, compression bits or source selection on any old disclosed v79/v81 heldout TEST. Reuse exact v81 source-generation method only.

## Frozen source reconstruction before opening v82 TEST
1. Original fully pretrained pinned `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, original CB59 source artifact from CI v59 run 41507201766.
2. **Unchanged original v76 TRAIN**: collect real pinned original FP32 teacher and original C-native packed 135M student logits/hidden state for original WikiText+AGNews TRAIN FIT/VALID; regenerate parent v76 paid rank4 snapshot. Never move a trained checkpoint after evaluating v82 TEST.
3. **Unchanged original v81 TRAIN**: same original SciQ TRAIN `1050:1082`, BoolQ TRAIN `2500:2564` and QA TRAIN VALID SciQ `1180:1204`, BoolQ `2700:2732`. Real C-native base compressed hidden trajectories; same `seed=81`, Adam `lr=.003`, hard per-step backtracking `α=1,.5,.25,.125,.0625,.03125,.015625,0` to keep WikiText/AGNews original TRAIN-CAL each within parent +0.25 KL nats. Fixed checkpoints `[0,4,8,12,18]`, promote at least +5pp QA on one task and no QA regression on other, plus both TRAIN-CAL KL safeguards, otherwise retain original v76. Execute identical source TRAIN optimizer in two independent CPU processes on same host; verify exact original-trained source SHA, bitwise repeat, all physical source bytes, archived source CRC/SHA **BEFORE** opening v82 TEST.
4. Compare newly regenerated v81 source SHA to previous verified v81 run 41555699743 source SHA **`94282c492713b053ccabc76510bad5f45791142866aa81abad62b35081e752c0`**. If cross-host differs, **report** divergence; do NOT alter fixed source criterion, silently substitute another checkpoint or tune to v82 TEST. Cross-host bitwise reproducibility was already imperfect in v76/v77.
5. Run exactly four sources as actual whole original 135M pretrained models: separate FP32 pretrained teacher, original CB59 native 5bit, frozen parent v76 paid CB76 and frozen v81 TRAIN-selected paid CB76. Every original trained source 2D matrix remains packed direct C in all compressed students. No original pretrained dense 2D weights loaded in hot student.

## BRAND-NEW larger original independently heldout TEST (fixed before training)
- Original SciQ **official TEST indices [160:240)** = **80** originally planned source questions; original support passage first320 tokenizer tokens; full original answer-option **length-normalized teacher-forced log likelihood** (no one-letter prompt), with uniformly source-based removal of duplicate/invalid choices *before* any score and exact excluded row indices recorded. 25% random-choice chance; original teacher must do reasonably well to validate diagnostic.
- Original BoolQ **official validation [2990:3070)** = **80** original labeled question rows, first256 passage tokens and unabridged question, exact single-token Yes/No completion. Original pretrained teacher's performance anchors usefulness; not an official untruncated BoolQ benchmark. Official split has >3070 rows. None of these examples has appeared in prior CALIBRI experiments.
- Original WikiText2 official `test` tokenizer tokens **[140000:141024)** = **16 independent 64-token contexts / 1008 scored next tokens**, distinct from all previous windows. Actual FP32 teacher→source full-vocabulary KL, actual original next-token NLL, original token SHA, paired window bootstrap 95% CI and CPU prefill.
- Original AG News official `test[7410:7510]` text-only first1024 tokenizer tokens = **16×64 contexts /1008 scored positions**. Official AGNews test split has 7600 records, and previous v81 `7200:7300`, v79 `6000:6100`, etc are disjoint. Token and dataset-revision SHA logged; no labels used for language.
- NEW fixed actual 4×24-token greedy KV-cache prompts, preregistered:
  `If the results are consistent, the next explanation is`;
  `The new discovery surprised the researchers because`;
  `The key difference between the two approaches is`;
  `At the beginning of the story, the traveler noticed`.
  Distinct-token share, largest-token concentration, post-prefill CPU token/sec and exact source SHA per model, with honest post-EOS fixed-length microbenchmark caveat.

## Independent confidence and go/no-go rules — scientific analysis only
**No change to weights/checkpoint/parameterization can be made because of this TEST.** The original source is frozen before all v82 TEST scoring. Additional output diagnostics compute each model's raw correct count, error cases, pairwise EXACT two-sided McNemar p and 5,000 original-question paired bootstrap resamples for accuracy deltas v81 vs v76, vs CB59 and vs original FP32 teacher. Formal caveat: 80-item single-slice multiple comparisons and teacher-slice imperfections; do NOT claim significance at alpha .05 without careful multiplicity adjustment.

Predeclared post-hoc *research go/no-go* checks (cannot promote model retrospectively): supported original teacher SciQ ≥60% correct; v81 SciQ correctness not lower than parent; v81 BoolQ ≥+10 percentage points vs parent, and paired bootstrap CI lower bound >0; independent wiki and news KL v81-v76 regressions each ≤0.25 nats; v81 greedy distinct-token ratio ≥0.25. All must pass before interpreting v81 as a promising generalizable source prototype, and even then broader model/QA/generation evaluation needed. If any fail, v81 remains a **research-only negative/mixed result**. This threshold is separate from original TRAIN optimizer selection.

Physical source budget **≤0.700 bpw including all tensors, aliases, header, CRC and SHA** is mandatory. There is no new network, training data, or sidecar in the source; evaluation datasets and scores remain temporary CI audit data.

## Reporting
Only after successful GitHub Actions should original real four-model results be published and negative effects disclosed. Source physical metadata is recorded. Neither Jessica chat is tested nor production updates deployed.

**Status:** Preregistered, awaiting actual GitHub Actions test result.
