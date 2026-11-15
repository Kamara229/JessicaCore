# NRA2 3.0 S4 — Russian grammar and semantic TRAIN contract, no eval

## Why

The original Calibri 2.0 / NRA problem is poor *generalization* rather than a missing collection of example answers. NRA2 3.0 therefore retains **one unified neural area** with 4 grammar layers, 3 semantic layers and 6 relational rounds. S4 prepares a real Russian UD-style TRAIN-only corpus, wordpiece/token-to-word alignment, and a differentiable grammatical/semantic objective. It is not an external grammar adapter. The original 43M model, Green Tunnel, 616 J7 checkpoint and all other original routes stay frozen.

## Key architectural obstacle solved in the objective, not yet empirically

Russian words can precede their syntax heads, including subjects preceding verbs. The original S0 `dependency_logits[t, j]` correctly masks future tokens, but cannot *by itself* predict the head of a word that lies to its right. Disabling its mask at time t would leak future text. S4 instead defines a separate **completed-prefix readout** *from the existing NRA3 layers and heads with zero added learned parameters*: at the final **currently visible** token, contextualize every earlier word representation using the current final-token state. Head predictions may then point to any already-visible word, including to the right, plus a pseudo-root, without seeing future generated tokens. It is a training/inference objective at a completed available prefix, not clairvoyance at intermediate steps. No persistent KV semantic cache or true text decoding is claimed yet.

## TRAIN source and boundaries

The pilot has 12 annotated genuine Russian sentences with subjects, objects, indirect objects, source obliques, changed word order, adjective-noun links, and passive voice. Every sentence is exclusively `NRA3_S4_RU_UD_TRAIN_ONLY`, and source, annotations, morph tags and distinct exact text hashes are included in the prereg report. Russian subject is **not** automatically semantic agent: "Павел получил письмо от Анны" has subject=RECIPIENT, Anna=SOURCE; passive grammatical subject is PATIENT. Pronouns/coreference, complex nested clauses, temporal inference and general Russian grammar remain out of scope of this restricted first corpus.

S4 never accesses the previously consumed five independent EVAL128 sets or the unopened sixth S3CI EVAL128; there is no QA/test answer, graph reachability or directional gold. No external heldout file or ordinary Jessica chat query. These twelve examples are already development TRAIN and must **never be reused as independent unseen EVAL**. Their labels can be used for supervised training only after source identity and tokenizer/hidden320 SHA are separately locked.

Fast tokenizer's offset mapping must agree byte-for-byte with actual Calibri prefill token IDs; one token may not overlap two distinct words. Word-supervision uses each word's last subtoken. A fake character-level tokenizer is used strictly for offline contracts; **real Calibri tokenizer and original true 43M layer-8 CPU hidden states have NOT been provided in S4.** S4 objective is checked for differentiability on mock hidden320 only; **NO actual linguistic training has been performed**, and no FP32 weights exported.

The same finalized-prefix scoring contract is checked against native C CB59 packed inference of the same quantized weights. Reuse existing UPOS, morphology, dependency and semantic heads in NRA3. The relational 6 rounds have no supervised target in this S4 loss yet; supervised semantic graph links are a separately reviewed next step. Full permitted 2048 TRAIN physical SHA denylist, real Calibri features and training-aware physically paid CB59 QAT require a future gated stage.

`output_gate=0`, `candidate_accepted=false`, no actual real-model 616 J7/EOS or CPU >=90%, no complete CALIBRI+GT source physical <=0.7 proof. Remain research-only, DRAFT, DO NOT MERGE.
