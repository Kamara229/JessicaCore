# S2 — causal failure localization + **physically paid** GT capacity study (no NRA)

**Important:** do not infer pretrained FP32 parity or claim working Green Tunnel before meaningful generated answers improve. S0 135M B loss, S1 4-cycle 0/12 and architecture C0 0/12; these are the actual failures to investigate.

## Frozen decisions

1. Diagnose compression distortion from the **original pretrained** 134,515,008 FP32 SmolLM2-135M-Instruct checkpoint (exact pinned revision), not retrained random weights, against the original independently compressed S0 physical archive, without model training.
2. On four **already inspected TRAIN/debug prompts** only, record original FP32 vs direct CB59 model input embedding and every original 30 decoder hidden state: last-token cosine, relative L2 norm, top-5 logits, FP32 argmax rank in compressed; 8-token greedy generations, literal EOS, first token equality.
3. A/B interventions are **DIAGNOSTIC ONLY / UNPAID, NEVER eligible as a model**: temporarily replace only the compressed input embedding with the FP32 teacher embedding, temporarily replace only the compressed output head (tied original source) with FP32 head, both, then restore original CB59 modules in memory. This separately localizes tied embedding failure vs internal-attention/MLP failures, not a solution inside 0.7 bpp.
4. Benchmark real CALIBRI 2.0 **scaled architectural capacities**, physical FP16 serialized sidecars + source + alternative S1 delta + manifest. No new full S1 duplicate, no dense teacher/diagnostic hybrid in budget, no NRA/S3:
   - baseline 2 local GT×rank8, recurrence rank8, slot rank16;
   - 5 local GT×rank16, recurrence rank16, slot rank16;
   - 4 local GT×rank16, recurrence rank32, slot rank16;
   - 5 local GT×rank16, recurrence rank24, slot rank24;
   - 5 local GT×rank24, recurrence rank32, slot rank16.
   Test all 30 GT recurrent hooks via synthetic forwarding with exact pre-training identity and actual nonzero gradient through recurrent gates; never mistake untrained neural blocks for quality recovery.
5. Count **every** sidecar parameter and physical serialization byte at ORIGINAL unique-parameter denominator. Report all-in bpp with/without S1 profile bank (S1 previous 0/12); reject above 0.7. Capacity is a budget exercise, not quality.
6. No new unseen/heldout EVAL yet: reserve that for after training/selection. No silent inspection of sealed S118/previous QA.
7. Production and other CALIBRI/NRA branches untouched. Only research PR, GitHub Actions on CPU. Ordinary Jessica chat is not used for testing.
