# NRA2 3.0 S5 — authentic Calibri 43M layer-8 hidden, read-only

**Separate DRAFT research branch. Do not merge; output_gate=0, candidate_accepted=false.**

S4 locked 12 Russian UD TRAIN-only sentences (canonical full annotated SHA `6ee1f270d413f4cfc79727bd48386ee29881520fcf6834309869733fd00f7a61`). S5's single aim is to obtain their **actual frozen original CALIBRI 43M CPU hidden320 representations at the output of layer #8** under the original S93 / Green Tunnel / S103 all-layer routing, while preserving the precise S194 original 616 J7 generated token SHA.

Implementation follows successful earlier S3AZ real CPU source-capture proven flow, **not** a mocked 43M checkpoint. Existing original S194 `real_j7_baseline` executes first: 456 protected old TRAIN, 128 new TRAIN, 32 original S190 TRAIN probes. Only after those genuine generated outputs are captured does a new read-only hook register at layer index 7 *inside* the authentic frozen `frozen_cpu_phase_scope`, after the pre-existing GT/S103 hook registration. It collects 12 purely textual Russian TRAIN prefills. UD annotations/semantic roles are **never passed to the original Calibri CPU model**, and original full 616 predictions are subsequently checked for exact checksum/accuracy against immutable S194 physical reference.

The hook returns `None` so PyTorch uses the untouched original output object; no NRA3 writeback, original 43M/GT optimizer, sidecar, routing or LIVE changes. Captured original `[12,T,320]` FP16 layer-8 hidden, actual token IDs, right-padding masks, and lengths are physically saved with full SHA and exact readback. Real tokenizer's original exact chat template/fast offset mapping must match the CPU prompt IDs, with fail-closed cross-word merges. A JSON report contains only training source SHA, alignment positions, feature SHA, S194 generated token digests, and stage blockers. **No semantic-role ground-truth tensors, test IDs or QA labels are exported.**

This is still **not actual linguistic training** and not a full-answers+EOS gain, native CPU speed-ratio, packed CB59 0.7 all-in certificate, completed semantic caching or J7 candidate acceptance.

Workflow mirrors the proven S194 research-only physical source imports; prerequisite S4 provenance and real 616 J7 are SHA/run checked and the 12 training sentences are outside all five previously consumed EVAL128 and unopened sixth S3CI. A failed real tokenizer alignment or missing old artifact must halt S5 rather than bypass the gate.

Next **separate** stage may train the unified grammar and semantic heads on actual source only after verifying exact historical TRAIN2048 whitelist and independent denylist, and preserving the frozen original Calibri J7.
