# CALIBRI Pro Max — GREEN TUNNEL v121 architecture decision after actual v120 failure

Date 2026-11-04. **Design specification only; not a validated implementation or a released Jessica service.**

## Empirical reasons to move beyond text reranking

Actual 135M v120 independent 2,054,625-param 8,223,732-byte UTF8 critic trained on 34 human-labeled examples / negatives: 91.67% recognition of presented held-out gold answers but only 54.17% rejection of synthetic negatives; **accepted 16/16 genuinely unusable** autonomous v119 outputs including empty strings and repeated 'Yes'. Actual correct FIRST answer subtoken among model tokens:
- pinned CB59 top1=0/24, top32=0/24, top512=2/24
- v119 joint corrected top1=4/24, top32=10/24, top512=14/24.
First correct token is only an optimistic upper bound, NOT a full fluent answer. Output-only verifier lacks recoverable options. Scientific FAIL.

## Requested next architecture: active compact independent controller

Physical semantic/control network G_theta stored one time (target budget <=10,000,000 bytes **including** tokenizer, manifest, weights). It is a separate neural model with its own pretrained/trained weights, not a replacement for compressed CB59. It should accept:
- text of the question and partial response through common UTF8 byte encoder (independent from base vocabulary),
- standardized top-K entropy/rank/repetition stats from host logits,
- optional host-adapter *fixed-dimensional sketch* of hidden states;
and produce:
- a small bounded vector z_t (e.g. 16 or 32 floats),
- multiple confidence/stop/retry/action logits, and
- a scalar policy/gate based on normalized relative layer depth and typed operator role.
Do not suppose a fixed 576/1536 latent width for 10MB–10GB hosts; standardization requires module-specific adapter.

Integrate with v119 **paid existing rank1 operators**, not new full matrices:
    h_{i+1} = F_CB59_i(h_i) + alpha_theta(z_t, i/L, role_i) U_i (V_i h_i).
Thus controller changes the *path of computation before next token*, not just polishes final prose. U_i and V_i are already paid inside 0.697352789 bpp v119; only gates and controller are additional. For a future architecture with no existing U/V hooks, model-specific lightweight adapters MUST be serialized and counted. Carry state from previous token (causal), and only recompute controller when useful to meet <=10% real latency cost.

## Brain/route metaphor, grounded implementation

"Green tunnel" is not literally moving neurons into physical logical chains. It is a trained adaptive *feedback control policy* over activations and output probabilities. It can add coherence, grammar preferences, goal relevance, and selective correction when the host still contains informative internal features. It **cannot guarantee** to recover lost world knowledge from nonsense activations without enough new knowledge in its own 10MB weights, external verified data, or additional learning. Abstention on unrecoverable requests is a valid safer behavior, but not a correct answer.

## Data and learning

Build larger train-only corpus: math with verifiable arithmetic, multi-step instruction following with automatic checks, grammar and multi-sentence stories judged with dedicated rubric, factual tasks grounded in known reference documents, actual BAD outputs from CB59/v119 including empty/repetitive/generic single tokens, alternative well-formed responses, multilingual outputs. Train true quality/ground-truth objective, not teacher KL nor matching FP32 original weights. Random split by task *family*, not only literal prompt; independent unseen TEST remains closed until a frozen candidate.

Pilot future comparison:
A original CB59; B paid joint v119 (0.697353 bpp); C separate G_theta + only gating of already paid v119 operators; D separate G_theta + text critic (compare incremental advantage to C). E source model independent 14M Pythia and 135M SmolLM2 for actual transfer; a 10GB architecture remains unproven until tested. All models strictly report (1) core 0.7bpp, (2) **full system disk bytes**, (3) sidecar bytes once, and (4) CPU wall-clock / tokens-per-second including controller overhead. No backend/Render/production changes.

## Non-negotiable budget equation

Original unique denominator N=134,515,008, native v119 full source=11,725,552 bytes (0.697352789 bpp). A physically separate 10,000,000-byte controller gives a complete single-host total 21,725,552 bytes = 1.292081966 bpp. Actual 8,223,732-byte v120 critic gives 19,949,284 bytes = 1.186442125 bpp. Remains only 44,511 bytes under 0.7bpp on the **original 135M single-host** checkpoint, impossible to fit 8–10MB independently without recompacting/removing other weights. One shared controller across many models could amortize cost for internal accounting, but total physical resident memory/storage must still be reported truthfully. Separate architecture and product goals must not be silently conflated.

Next implementation should first verify gradient sensitivity/control authority of existing rank1 adapter *gate* with actual real 135M native packed autograd in v119 before spending larger experiments on standalone G_theta. Show that varying gates changes faulty continuation in a task-dependent useful way (not merely off-topic smoothness). Test answer correctness, multi-sentence coherence, EOS, repetitions, actual CPU, and full bytes. If gates can't recover useful semantics, must change source codebook allocation/training or augment verified external information. No scientific PASS or claims of universal LLM repair from architectural design alone.
