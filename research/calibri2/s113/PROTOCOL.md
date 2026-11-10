# CALIBRI S113 — first-logit reachability and safety margin, genuine S101 CPU

**Research-only.** This experiment does not train weights, create a candidate model, open once-inspected QA/TEST data, run Jessica chat tests or modify main/LIVE Jessica/Render/Supabase/APK.

## Motivation

S110 established that 103 of 456 original authorized TRAIN full answers are wrong: 69 **first-token failures**, 18 second-token failures and 16 last-EOS failures. S104/S109 real MPS training of global shared-GT gains did not improve full-response accuracy. S106 increase of existing gain strengths yielded a new correct response with one old response forgotten (x2) and two acquired with five forgotten (x8). Thus the **actual first-step greedy logit margin** is an essential trainability and catastrophic-forgetting diagnostic. A falling surrogate loss does not establish a changed greedy answer.

## Predeclared method

Rebuild unchanged physical S101 36M compressed CB61 weights with original GT3/9, original S87/S91 S94 physical bank, frozen question-only S95 router, original S82 controllers, and frozen S98 second-token copier. The four source artifacts are checked against their immutable S99 SHA256 digests.

Generate all 456 original authorized TRAIN question answers via the **actual S101 cached greedy decoder**. Separately score each same question's exact first decoder step via the original S101 true rule:

`logits = model(input_ids=full_chat_prompt,use_cache=True).logits[-1] + F.linear(original_first_controller(last_hidden), original_lm_head_source)`.

The independent first-token argmax MUST match the already generated actual S101 first token on **all 456/456 prompts**. Final full+EOS quality MUST remain 353/456 and first-token quality 387/456; otherwise abort.

For each TRAIN target gold first token (gold is used *only* offline for this diagnostic, never as an inference input), compute its rank in full output vocabulary, logit gap of the wrong predicted top1 to correct gold, or correct top1 vs runner-up safety gap. Categorize rank in 1,2,3–5,6–16,17–64,65–128,129–512,>512 and by **frozen S95 route and TRAIN family**. Check if gold first token was literally present among original prompt candidate positions, without injecting it as a decoder answer. Aggregate and report distributions only. No raw questions, answer token IDs, or per-identity oracle lookup tables in outputs.

## What to infer and what NOT to infer

- Correct gold rank among first-token mistakes shows whether a low-dimensional gate could plausibly change argmax at small margin, or whether the underlying compressed model lacks content discrimination.
- Very small *originally correct* top1-vs-top2 margins flag potentially fragile protected skills. It does not prove an update will damage those examples.
- S95-route breakdown distinguishes errors addressable by S93-only GT from S87/S91 errors that S111/S112 cannot modify by design.
- A target token being among prompt token candidates does not prove copying is the correct way to answer. Literal presence can be spurious, and the S98 dedicated owner-copy module acts at second token, not the first.
- Actual all-inclusive bpp and real complete autoregressive correct answers remain the only model candidate acceptance evidence; first-token gold ranks are **diagnostics**, not a model-quality metric or a teacher at inference.

The analysis may inform a future new experiment with differentiated first-token loss and protected margin safeguards. It cannot itself justify extending the physical model or changing production.
