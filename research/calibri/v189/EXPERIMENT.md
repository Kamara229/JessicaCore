# CALIBRI v189 — stronger original model capability before stronger-weight compression

v188 completed 192 historical public-development
SmolLM2-135M FP32 responses. **66/192** were conservatively
machine-confirmed, **57/192 ungraded**; paired binding/Yes-No
tasks were poorly differentiated by this small original model.
Historical DEV answer-pairs are NOT unseen accuracy tests and
raw text must be preserved when interpreting automatic scores.

Our proposed comparator is the independent, publicly available
HuggingFaceTB/SmolLM2-360M-Instruct checkpoint: a larger
instruction-tuned model in the same family. This is a
**hypothesis that the original model may be more capable**,
not a measured result. Its own actual revision SHA and
unique tied weight count MUST be resolved and frozen before
generating any response. Never use the 135M unique-weight
count or bit budget for the 360M original source.

Precommit full original historical 96 public DEV questions,
TWO original pre-existing direct/brief formats (192 actual
prompts) and 32 greedy generated answer tokens each.
Load the archived raw v188 original output once, verify
every question ID, prompt SHA, and style before comparing
any scores. The model/tokenizer are loaded from the
frozen original 360M revision returned by Hub model_info,
then one prompt per exact v188 question. Use existing
conservative answer parser unchanged, report 360M and
135M per-domain machine confirmed, parsed, ungraded,
EOS, token diversity, and true opposing-pair behavior.
Never compare numerical token IDs across different vocabularies
without verifying tokenizer identity: compare actual literal
answers and each model's self-consistency instead.

Any positive difference is meaningful only for that small
non-independent historical set. Original FP32 135M answers
have many not-machine-graded cases; a native semantics-aware
human review would be needed for robust accuracy estimates.
No quantizer fitting or source selection uses these DEV
results. No source weights are compressed or changed in v189.

Proceed to a separate truly NEW public blind QA suite later.
Only after measuring the original 360M capability should
we decide whether to allocate the <=0.700 physically
paid bits/original-unique-weight model budget and
native CPU target >=90% to a new source.

Independent v89 private TEST remains unread; no
NLL/KL/PPL as success gates; no main, LIVE Jessica,
Render, Supabase, APK or Green Tunnel code changes.
Jessica merely relays messages and runs no tests.
