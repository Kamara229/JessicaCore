# CALIBRI S117 — 340D MPS-gradient feasibility with explicit protected tasks

**RESEARCH ONLY. No new model is trained, serialized or accepted. No old QA/TEST is touched. No changes to main, LIVE Jessica, Render, Supabase or APK.**

## Confirmed source observations

S101 historical source 36M compressed CB61 and two protected S93 Green Tunnel donors, frozen S94 numeric profile bank, S95 prompt router, S98 neural second-token copier, and S101 KV cached CPU decoder achieved **353/456 full+EOS, 387/456 first** on the original authorized TRAIN.

S105 measured opposing gradients in 46 of 96 10D task-pairs and the strict zero-loss first-order direction collapsed to zero in only ten dimensions. S111/S112 added a 2×10×17 340-parameter prompt-conditioned dual-GT bridge, physically FP16 serialized under 0.7 bpp. Three real MPS S112 steps yielded **+1 new correct complete+EOS, −7 previously correct complete+EOS**, with one gained first token, zero first-token forgetting. The candidate was REJECTED, and exact 340 weights and full Adam rollback confirmed. S114 shrinkage / donor-only ablations all lost at least one original full correct answer, thus cannot be accepted. S115 attempts to decouple initial prompt KV prefill from later decode.

## What S117 measures

The question: **Are there nonzero first-order improvements for the same 340 coefficients that do not increase the loss of a wider protected TRAIN cohort?** In ten dimensions S105 could not find one; 340 prompt-dependent dimensions might admit more independent interventions.

- Validate original S101, S111, S112 SHA/provenance and real 353 full, 387 first outputs reconstructed from the frozen 456 authorized TRAIN examples.
- Select 8 currently WRONG-FIRST S93-routed TRAIN examples balanced across known TRAIN families. This uses only pre-existing immutable TRAIN labels.
- Protect all **7 exactly identified S101-correct full answers that S112 forgot**, plus 16 additional S101-correct S93-route TRAIN examples spread by family (23 per-example constraints total). Their identities were discovered entirely on authorized TRAIN. NO independent QA or per-question routing logic is created.
- On genuine Apple MPS, restore the frozen 36M original S101 compressed CB59/CB61 model, original trained GT3/GT9, controller, frozen S95 feature extractor. For each of 31 examples compute REAL autograd gradient on the 340 in-memory prompt-conditioned coefficient parameters at **zero initialization**, with teacher-forced S91 complete+EOS policy loss plus real initial S101 decoder-controller token-0 cross-entropy. **No optimizer step.**
- Apply a GEM-inspired Euclidean gradient projection of the average wrong-FIRST gradient descent direction onto 23 halfspaces where each protected surrogate loss has first-order non-increasing change. Compute in CPU FP64 with an iterative Dykstra convex projection, numerical fail-closed zero if constraints are violated.
- Report number of opposing new-vs-old per-example gradients, protected constraints violated by raw descent, projected norm fraction, projected new-loss first-order improvement and layer/donor decomposition. Do NOT serialize a new model or change training weights.

## Why this is NOT retention proof

Gradient Episodic Memory (GEM, Lopez-Paz & Ranzato, NeurIPS 2017) and conflict-gradient projection (PCGrad, Yu et al., NeurIPS 2020) motivate studying gradient interference but **first-order frozen surrogate loss protection is NOT exact greedy full+EOS protection**, especially near quantized FP16 output boundaries and downstream original S98 copy decisions. Projected directions may still lose old tokens once applied. The only way to accept a future learned proposal is an actual physically serialized S101 full 456 authorized TRAIN reconstruction with all 353 full and 387 first identities protected, 0.7 bpp physical bytes, >=90% original CPU speed, new correct full+EOS answers and independent once-only heldout validation. No raw TRAIN questions or output-token oracle is saved by S117.
