# CALIBRI v192 — frozen complete-answer grader gate PASSED (2026-11-05)

**Verified actual outcome:** [GitHub Actions #42094376648](https://github.com/Kamara229/JessicaCore/actions/runs/42094376648) **SUCCESS**. Six source-independent adversarial unit test groups completed in Python 3.12, no model imports, no weight files, no source generations. [Reproducible frozen grader manifest #13446881599](https://github.com/Kamara229/JessicaCore/actions/runs/42094376648/artifacts/13446881599).

- Version: `v192-locked-grader-v1`
- SHA256 of canonical sorted source-file manifest: `e230e480c02c680ae02d17b0825b451cbf76b138ceb743f181cc6fb62ad4ac1f`
- Sealed modules: `research/calibri/v192/contract.py`, `recognize.py`, `grade.py`.
- Adversarial unit tests: **6/6 PASS**, including full numerical equations vs different operands, negative or ambiguous gold wording, correct vs wrong folder-owner subject, symbol/country factual reference guard, and **both-correct** requirement for reversed Yes/No pairs.
- Model generations: **ZERO**; model or compressed weights changed: **ZERO**. This is solely infrastructure/measurement logic, not actual correctness of any model.

The fixed grader distinguishes `machine_correct`, `machine_wrong` and `ungraded` while preserving exact raw text, case ID, source of recognition and original question. The future scientific execution must independently freeze entirely **NEW** public question text, gold, sources and hashes, then pin THIS exact grader source digest before original model evaluation. Revise the grader only under a new version, and do not call post-hoc changes an unbiased accuracy improvement.

**Historical controls remain scientifically distinct:** v190 preregistered original FP32 135M/360M bare-answer strict scores **9/64 vs 18/64**; v191 exploratory after-answer structural rerating **34/64 vs 43/64**. The new v192 grader has NOT been independently tested on new model outputs and cannot retroactively supersede those original measurements. This research is still **pre-compression**: original 360M 361,821,120 unique model parameters, a hypothetical <=0.7bpp complete file budget 31,659,348 bytes. There is no measured 360M CALIBRI archive, 360M native-speed or compressed QA parity here.

Scientific next step: build a fresh fully source-frozen and **verifiably disjoint** 64–128 new public QA suite; use original 135M/360M FP32 only as controls with the v192 SHA. If those agree with a stronger 360M source, proceed to design and run actual source-paid quantization experiments under 0.700 bpp and CPU >=90% before comparing compressed outputs. Preserve negative pair-domain results, don't optimize on public test golds, leave sealed private v89 TEST untouched.

No live Jessica, main, Render, Supabase, APK, Green Tunnel or private source changed. Research-only draft PR, unmerged. Jessica model text tests never requested.
