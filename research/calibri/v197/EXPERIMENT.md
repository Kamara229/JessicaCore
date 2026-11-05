# CALIBRI v197 — Green Tunnel source engine and HARD FP32 parity gate

## Mandatory owner constraints

**NO scaling beyond ORIGINAL 135M** until a physically compressed CALIBRI 135M answers the same original questions as the original pretrained FP32 135M, without systematic full-answer regressions. Actual matched native CPU throughput must meet the earlier >=90% CB59 gate, and the entire archive must stay <=0.700 bits per original unique parameter. Training losses, KL/NLL/PPL, correct first tokens, SHA/CI passing alone, post-hoc answer parsers or comparisons with 360M may not replace real generated answer correctness.

Goal A: evolve GREEN TUNNEL into a reusable weight-restoration algorithm and explicit codec adapters; larger future models may need more recovery cycles, but more cycles are a tested hypothesis, not a guarantee.

Goal B: continue actual source training from the already partly restored v186 135M compressed checkpoint, without discarding its physical learned codebooks, and improve real full answers.

## Actual baseline and measured negative results

- The real previously trained v186 SmolLM2-135M physical file is 11,229,303 bytes = 0.6678394131 bits/unique parameter, SHA 933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a; native CPU speed 99.59% of original CB59; 2/24 confirmed full historical DEV replies, 3/12 held-out contrasting first-token pairs. Genuine partial but extremely weak quality.
- v194 360M physical 0.648907 bpp had **0/16 machine-confirmed full answers** and gibberish. This scale experiment was diagnostic; no more bigger-model work authorized.
- v195 C-numerical decoder agrees with exact CB59 oracle (max relative L2 4.45e-7 on sampled matrices). Six selectively unpaid FP32-family repairs all gave 0/8 machine-confirmed. Recovery should be joint across the network.
- v196 is training real authenticated v186 135M via 2 cycles x 96 steps, gradients to all 211 paid source codebooks from full 49,152-token teacher output distributions at TRAIN time positions, mixed with balanced opposing TRAIN pairs. v196 CI results must be observed, not assumed.

## v197 universal core (not a performance claim)

The green_tunnel source CONTRACT reads ACTUAL full physical source file length, verifies whole-file SHA256 and tied alias references, and rejects any archive above <=0.700 bpp or a new uncharged neural sidecar.

The green_tunnel CB59 adapter reads exact unique physical trained tensor metadata, dynamically discovers its matrix family/depth and builds a common Operator list without a hardcoded SmolLM2 211 count. An alternative LLM/MoE still requires a proper separately tested codec and model adapter.

The schedule engine gives EVERY source operator a chance for discrete index recovery EACH cycle; it interleaves transformer depth/family, rather than allowing a fixed v186 subset of only 96 matrices to monopolize source code swaps. All original paid codebooks can continue to receive whole-model gradient on all steps.

The parity evaluator uses the EXACT original frozen whole-response grader on both original FP32 and compact model; requires the same original case IDs, retention of each FP32-confirmed answer, no worse domain correctness or ungraded count, both-correct reversed pairs, valid native EOS/diversity, complete physically paid <=0.700 bpp, >=90% matched native speed, and a sufficient independent untouched >=96-question set. Historical 24 and v193 16 DEV questions cannot authorize scaling, even when their scores tie.

The original v192 scorer is incomplete for free-form factual sentences, so machine-confirmed parity is necessary but is not itself proof of human semantic equivalence. More diverse completely new presealed QA and human checks are needed before any real parity conclusion.

## Immediate real output and next algorithmic cycle

v197/audit_v196.py will inspect the actual completed v196 physical artifact and its 16 real new-development replies against the previously sealed 135M FP32 answers, preserving EVERY output including failures, repeats and incomplete scoring. The next training iteration should use the dynamic complete 211-operator index-update schedule, continued full-vocabulary source-only gradient restoration, and physically serialized paid CB59 source on each completed cycle; use ONLY TRAIN for intermediate decisions. Reserve untouched questions for independent final certification.

All work isolated DRAFT research PR; no merge, production Jessica/Render/Supabase/APK or private v89 TEST, and no Jessica-chat test.
