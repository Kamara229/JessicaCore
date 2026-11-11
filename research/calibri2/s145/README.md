# CALIBRI 2.0 S145 — Neural Skill Blocks (research only)

## Decision
We **retain the ~135 million parameter architecture goal**, but grow by adding
independently trainable, versioned neural Skill Blocks instead of expanding a
shared adapter blindly. The current experiment is a **CPU-only structural
preflight**; no claim of language progress, production acceptance or parameter
growth on an actual checkpoint is made.

The guiding analogy is Jessica's versioned Experience Skills and atomic updates,
but the neural Skills here are real learned matrices and activation gates,
**not** a library of exact prompts or stored answers.

## Verified parent (source only, not deployed)
- S142 macOS run `43078895081`: one protected J7 transaction committed,
  historical TRAIN full+EOS **361→362/456**, S122 newer TRAIN **17/128**,
  zero inherited first/full losses, physical reported bpp **0.6568244742**,
  paired CPU ratio **0.98869**. Independent unseen QA **NOT RUN**.
- S144 independent preflight run `43090174427` verified the S142 source
  archive, FP16 checksum and tampering controls. No model training there.
- S144 continued J7 run `43090174216`: 0 gains, 1 exact rollback; its
  workflow marked failure due to GitHub upload-artifact HTTP/HTML error.
  **Do not** treat the S144 no-gain checkpoint or failed artifact as an
  improved or accepted parent. The immutable S142 source is still the reference.
- Source pin is recorded in `research/calibri2/s144/source_manifest.json`
  and already verified by S144 source-checksum CPU tests. This branch is
  based on the S144 *research branch*, not on main or LIVE.

## What is implemented in S145 CPU preflight
- `NeuralSkillBlock`: low-rank down→SiLU→up residual with exactly zero
  initial output, so attaching a Skill does not change original outputs.
- `NeuralSkillLayer`: separate learned per-token gate and residual for
  each Skill. Causal gate never reads future tokens and never uses prompt
  equality, QA labels, IDs, answer dictionaries or retrieval at inference.
- `LayerwiseNeuralSkillBank`: attach Skill to several layers, select one
  trainable Skill, freeze every inherited block **and** its gate, version
  Skill metadata, and account for ALL actual tensor bytes and manifest bytes.
- Eight deterministic CPU contracts: identity insertion, invalid IDs,
  frozen old weights, real gradients, causal routing, two-path composition,
  stable versioned manifest, provisional FP16 byte accounting.

**Limits:** no real S142/43M weights are loaded here; this module is not
yet inserted into the true S101/S142 KV-cache prefill, active S82 neural
controllers, second-word pointer and EOS decode. Passing these synthetic
CPU tests does **not** demonstrate new answers, compression acceptance,
native-C speed, or transfer to novel tasks.

## Growth to 135M: count what is actually learned
Target milestones (tentative): 43.26M → 50M → 65M → 85M → 110M → 135M.
Each step must separately report:
1. architectural parameters, nonzero learned parameters and actively used
   parameters (do not count empty zero-filled blocks as acquired skills);
2. physically serialized packed model, all adapters, gates, metadata, routing
   and manifests: total physical bytes and **all-in bpp ≤0.700**;
3. paired reconstructed CPU speed **≥0.90** of agreed baseline;
4. full-generation protected TRAIN **456 historical + 128 S122**, including
   first-token identity, all answer tokens and EOS, no forgetting;
5. genuinely unseen, preregistered multiword/compositional QA — never used
   to train, select or tune checkpoints.

A single rank-8 Skill spanning layers 3 and 9 at hidden size 320 has
**10,882** parameters (block + gate); in plain FP16 that is 21,764 bytes
plus manifest. For ~43M parent, the reported S142 bpp leaves only about
**233 KB** at the 0.7 cap before accounting for any changes. This is why
many Skills at 135M require new genuine physical codebook/bit packing,
router sparsity and verified CPU throughput; merely allocating FP16 Skills
does not satisfy compression.

The count 135M is a **capacity target**, not a permission to create unused
filler parameters or to drop inherited answers.

## Next executable scientific experiment (separate guarded research step)
1. Reconstruct immutable S142 FP16 and assert true 456+128 full/EOS parity.
2. Insert the first small Skill at candidate layers 3/9 in **actual**
   S142 hidden-state computation and differentiable full-policy S101
   KV-cache + first/later controller + pointer + EOS training. Freeze all
   inherited base and original Skill paths; prove gradients across both
   down and up and gate.
3. Serialize every trial to a physically reloaded packed CPU model and
   perform exact atomic J7 commit/rollback of weights AND optimizer. Require
   no full or first-token losses, all-in physical bpp ≤0.7, CPU ≥0.9.
4. Only after first safe new TRAIN improvement, train a second independently
   versioned Skill with Skill 1 fully frozen. Compare to a single adapter
   with the same *measured* trainable parameter and byte budget; perform
   ablations and check compositional novel answers.
5. Freeze model/architecture/hyperparameters before preregistered, one-time
   independent held-out QA. Older inspected QA remains embargoed.

This branch contains only research Python and a CPU workflow. No changes to
`main`, Jessica LIVE, Render, Supabase, Android APK or Jessica chat tests.
Keep PR DRAFT, no automatic merge or promotion.
