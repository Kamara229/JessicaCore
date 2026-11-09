# S3 rerun provenance and bounded TRAIN-only next step (2026-11-09)

This note reconciles two **different successful runs**, not a single reproducible result. Do not replace one run's numbers with the other's or select a checkpoint based on held-out results.

| GitHub run | step 1920 exact full TRAIN+EOS | step 2560 | physical bpp | packed SHA-256 reported by process |
|---|---:|---:|---:|---|
| 42496717434 | 127/176 | 124/176 | 0.6910469807298895 | e4d9a4ef7705c660b803fd823263d0eceb804b4732afdfac30a38884fc96d159 |
| 42530436760 | 125/176 | 122/176 | 0.6910469807298895 | b7f6450e29acfa53699314392853efae1029bfc158d30d3d62a06767905d8720 |

Source of second row: run 42530436760, job 128883234693, events `CALIBRI2_S3_FIXED_TRAIN_MILESTONE` and `CALIBRI2_S3_TRAIN_COMPLETE`. At steps 1920 -> 2560: old numeric yes/no 12/12 -> 11/12; arithmetic 10/12 -> 8/12; comparison 26/32 -> 29/32; two-word 3/32 -> 1/32; sentence 2/16 -> 1/16. Thus an overall score obscures opposing retention and multi-token degradation. Reloaded physical checkpoint also scored 122/176. The archive digest sha256:61fa2d2a7d510e9f753a049c424360eb15b5ca998f20acdb3e86f3b0f7de3fb7 is a **ZIP digest, not the packed-weights digest**. Independent packed-byte SHA and comparable native CPU speed >=90% are not yet verified.

## Precommitted low-cost diagnostic, before any new training

1. Download the existing TRAIN artifact only. Check ZIP digest, list files, independently hash the packed binary and verify its exact physical byte count, architecture parameter count, 43 matrices, and paid CODEBOOK+5bit <=0.700 bpp. Compare to the process-reported packed SHA. Fail closed on any mismatch.
2. Run `research/calibri2/s0/audit_train_answers.py` on the saved TRAIN JSON; capture every pair's two sides, full sequence including EOS, unique first-token count, wrong-first-token and repetition flags. Record the actual JSON path and analyzer output SHA. Token repetition heuristics are not semantic repetition. If the analyzer fails, fix only the research S0 analyzer and its test fixtures; do not fabricate metrics.
3. Before a new training pilot, record decoded text, exact EOS placement and token IDs for every TRAIN example, including both sides of each opposite pair. Compare *fixed* milestones, not best-of checkpoints.
4. Only if audit integrity passes: predeclare one bounded 22M pilot with balanced replay of both opposite sides, old skills and multi-token examples. Fix steps, seed, replay weights and final checkpoint in advance, with no hyperparameter sweep or QA/DEV/VALID/TEST use for training, selection or stopping. Evaluate TRAIN retention by domain and complete EOS; do not scale up until robust useful generation.
5. Benchmark native CPU on the same machine and workload as the reference; report tokens/sec and ratio, never assert >=90% without a measured comparable baseline.

No new CI launched by this note. Do not touch main, LIVE Jessica, Render, Supabase, APK, legacy 135M v001 or 360M.
