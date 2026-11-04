# CALIBRI v103 — require correctness, not just teacher imitation

v102 Actions #41746831993 TECHNICAL PASS; separate complete same-byte
candidate 11,229,303 B /0.667839 bpp, unchanged original CB59 native
5-bit K32V8 decoder. Original tied head weight relative RMSE 0.6625156,
source-only codebook re-encode + 16-bit row-scale LS 0.6482970
(2.15% reduction), measured paired native speed ratio ~0.9992.
But actual candidate natural text still repeated garbage
('tempfile', 'elsewhere', 'regim'); whole-model native prompt FP32 top10
overlap 0/40 on ALL four prompts. CODEC-only top10 improved 1/40 to 6/40
with true teacher hidden states; quality clearly not restored.

**Critical validity discovery:** original pretrained FP32 itself
incorrectly answered naked, NON-CHAT-TEMPLATED synthetic prompts:
'16 - 3 = 27' (true 13); 'capital of Switzerland is Geneva' (true Bern).
Therefore matching teacher logits isn't sufficient. SmolLM2 is an
instruct/chat model; its original tokenizer chat template must be
used in next experiment and teacher quality compared to *actual gold*.

v103 is a **NO-TRAINING and NO-SELECTION** audit of already frozen
real v102 paid candidate vs bytewise frozen original CB59 vs original
FP32 teacher, on **new actual original BoolQ train[8780:8804]**
with independent ground truth. Use the genuine pinned model chat
template (role user + add_generation_prompt). Score constrained
next-token original truth Yes/No, show exact label balance and
per-row paired correctness for all three models. Separately generate
32 actual original KV-cached tokens from 4 NEW exact labeled
arithmetic/geography user prompts; show actual text, rough exact-label
prefix and human review requirement. Never conflate original FP32
prediction with gold label.

Candidate archive, original complete source, original full pretrained
reference and original v102 manifest must all be SHA verified. v102
file is downloaded by pinned historical GitHub Actions artifact,
not rebuilt or retuned with v103 evidence. Original unique source is
0.667839 bpp and no extra source/model bytes. No model selected
from this check, official v89 independent TEST untouched, no
Live/Jessica main/Render/Supabase/APK changes.

Facades v103/spec.py, qa.py, text.py, pipeline.py; launcher,
isolated unit tests and GitHub workflow. Technical SUCCESS is not
scientific quality PASS. Stop claiming practical progress until
correctness/actual coherent answers recover.
