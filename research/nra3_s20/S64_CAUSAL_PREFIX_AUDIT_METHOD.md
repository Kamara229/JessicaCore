# NRA3 S64 — native BPE / quote generalization audit (research only)

**No trained parameters.** Compare previously frozen S62 and S63 on a separate, newly preregistered 32-case author-written challenge, not on spent S63 HELD32. Original Calibri43M+Green Tunnel and S26/S57/S59/S60/S61/S62/S63 paid weights are immutable; output gate must remain zero.

## First-read causal separation

1. Freeze new S64 RAW texts and separate manual GOLD *in different Git commits*, before any S64 true-original CPU capture. Both Git blobs and commit SHAs are pinned in CI.
2. Check S64 RAW is distinct from S53..S63 and never load S64 GOLD into any CPU/model source-only tool.
3. Reproduce the **unchanged original** 616 protected J7 generated-token digest on the actual 12-layer Calibri43M+GT CPU, then read genuine layer8 native320 for S64 RAW32.
4. Compute genuine fast-tokenizer raw-word BPE fractional overlap, shared-native-piece unsafe masks, and an additional **SOURCE-ONLY token-prefix stability audit** at each original Unicode word boundary. Prefix stability can flag an issue, but even all-stable prefixes are **not sufficient to certify full-context hidden320/GT autoregressive causality**.
5. SHA-validate all archived weights and computed native339 without mutation. Run unchanged frozen S62 and S63 first on the same 32.
6. For fair, non-oracle comparison, apply **the same** unsafe/unclosed-quote mask to BOTH S62 and S63. Evaluate a still more conservative shared-prefix mismatch fail-closed gate to BOTH. This guards against unsafe evaluation using original fractional shared-BPE pooled vectors. Record S63 ungated ablation as explicitly **unsafe, non-deployable diagnostic only**.
7. ONLY AFTER all frozen predictions, import preregistered manual GOLD once, score exact predicate, directed RECIPIENT and REJECTED, per family, with all source/tokenization restrictions. NEW32 then becomes permanently SPENT; never retune and reuse as fresh held.

The 32 are human **author-written by a single researcher**, intentionally structurally diverse (nested quotes, indirect reported speech, unclosed quotes, multiple negation scope, nontransfer dative). They are **not** randomly drawn, independently authored, or representative. A high F1 on this set alone cannot justify release.

## Limitations and acceptance

- Research-only, no LIVE, Render, Supabase, Jessica, APK or original model modifications.
- No protected S38, S47, S48, EVAL32/128.
- Strict native row abstention and source-only prefix safety indicators do **not** demonstrate true online/full-context causal safety.
- Protected 616 J7 is verified on **unchanged baseline**, not an active S63/S64-integrated candidate.
- No active integrated J7 retention, <=0.7 bpp, CB59 >=90% CPU, or end-to-end streaming certification.
- `candidate_accepted=false`, `activation_allowed=false`, `output_gate=0` even if exploratory metrics improve.

GitHub Actions: `.github/workflows/nra3-s64-original43m-gt-new32-heterogeneous-prefix-first-read.yml`.
