# CALIBRI S100 — frozen S99 on one new set (research only)

S99 independently finished: **353/456** correct full+EOS on previously used 456 TRAIN; **zero** lost S95 full answers (287) or previously protected S87∨S91 282 answers/first tokens; **0.699573397602 bpp** conservatively charged including numeric copy network; paired CPU speed **0.942799556891** of unmodified S95 for the first 48 cases. This is real physical compressed CPU generation, **NOT** evidence of generalizing to unfamiliar questions. Candidate remains unaccepted.

Scientific falsification risk: S96's one-time set, which is no longer eligible for tuning or retesting, yielded **2/64** correctly copied new prompts and ratio **0.899747** vs S82. S98's 64/64 heldout TRAIN conditional second-word success may be confined to its format. Therefore S100 uses only frozen S99/S95/S82 exact weight artifacts with no new training, and predefines 64 new prompts using a candidate noun list **disjoint from every S82.NOUNS noun** (including every old S96 noun), with **two different new textual instructions** and new owner names.

Hard evaluation: compare actual full+EOS CPU generations from S82, S95 and S99, first-token and second-token correctness, strict novel regressions, physically recorded S99 copy activations and paired CPU speed. Do not use previous QA, old S96 prompts/answers, or leak novel results into any training or rerun. A nonzero S95→S99 loss or speed below 90% original S82 flags this developmental evaluation as failed. Even passing S100 does NOT promote a model: untouched sealed final QA, native-C parity and GPU parity remain.

No LIVE Jessica, Supabase, Render, main or APK changes. No Jessica ordinary-chat tests.
