# CALIBRI 2.0 S103 — Green Tunnel between every transformer layer (research only)

## Decision / motivation

The owner proposes attaching Green Tunnel to all 12 decoder layers so the
hidden state sent to the next layer is adapted by Green Tunnel. This is a
scientifically plausible mechanism to repair errors upstream of first- and
second-token generation. **It has not yet been shown to improve real answers.**

Checked prior stage S102: the validated S101 compressed candidate already
uses independent trainable low-rank Green Tunnel blocks on decoder layers 3
and 9, while the S95/S98 runtime retains S87/S91 alternate profiles.
S101's measured 456-TRAIN result is 353 strict full+EOS successes, 387
correct first tokens, 0.699573397602 inclusive bpp and 1.39158x S82
cached CPU throughput. Available physical margin is only ~1,940 bytes.
S102's initial joint training changed 353 to 348/456; rollback succeeded.

## Architecture: reuse paid tensor weights rather than buy 10 new adapters

```text
embedding
   |
decoder 0  -> shared GT_3 * tanh(gain[0]) -> decoder 1
decoder 1  -> shared GT_3 * tanh(gain[1]) -> decoder 2
decoder 2  -> shared GT_3 * tanh(gain[2]) -> decoder 3
decoder 3  -> original dedicated GT_3      -> decoder 4
decoder 4  -> shared GT_3 * tanh(gain[4]) -> decoder 5
...
decoder 9  -> original dedicated GT_9      -> decoder 10
...
decoder 11 -> shared GT_3 * tanh(gain[11])-> final norm/head
```

These are RESIDUAL pathways. The original decoder output remains present:
`h_(i+1) = h_i + tanh(gain_i) * GT_3(h_i)`, and the following
decoder consumes `h_(i+1)`. The existing dedicated blocks 3/9 still operate
normally. This does not create a second trainable 36M core or any dense weights.

The single existing layer-3 low-rank decoder is REUSED ten times. Its
weights are stored once. Ten new FP32 gain values (40 raw bytes) are zero
initialized. Thus the existing S101 path is an exact baseline until tuning.
The real increment in `safetensors` plus manifest is >40 bytes; compute
that increment physically, not from the tensor's element count.

`shared_all_layer_green_bridge.py` provides:
- Hook installation on every *missing* layer, alongside existing 3/9 hooks.
- Disabled by default; a route-scoped context manager which enables bridges
  only for S95 profile S93.
- Output tuple and KV-cache metadata preservation.
- Zero gain fast path under `no_grad`, without extra neural computation.
- Serialization of ONLY the new gain vector, not duplicate adapter weights.
- A removable research-only bridge module.

**This is an isolated architecture-only implementation. No GPU TRAIN,
CPU parity benchmark, quality TEST, or model promotion was executed.**

## Integration policy

On a loaded S101 model:
1. Restore S93 core, 3/9 tunnel, S94 bank, S95 router, S98 copy policy and
   S101 CPU KV-cache runtime from immutable physical artifacts.
2. `old_tunnel.install(model)` as before; then construct bridge with
   `SharedInterLayerGreenBridge(old_tunnel, total_layers=12)` and call
   `bridge.to(device).install(model)`.
3. Route each prompt with the existing S95 logic. For a single-model
   non-concurrent research worker only, run generation in
   `with bridge.profile(route): ...` so S87/S91 are never altered.
4. Use `bridge.get_bridge_state()` only for a proposed S103 sidecar;
   record the actual newly serialized physical bytes and SHA.
5. Call `bridge.uninstall()` and `old_tunnel.uninstall()` after the study.
   No live user-serving service may use a mutable profile flag.

## FIRST scientific steps — with strong guards

**A. Infrastructure/parity before training:**
At zero gains, compare S101 and S103 exact full+EOS token sequences for all
456 authorized TRAIN questions and for the same frozen S95/S98 routes. Check
same-run cold cached CPU throughput, KV-cache correctness, no profiler state
leakage and all-inclusive physical bpp <=0.7. A failed baseline invalidates
the experiment. Use the frozen S101 baseline, not exposed QA/TEST.

**B. Learn only the ten gain values initially:**
Freeze existing CB59 core, codebooks, 3/9 donor weights, S94 profile bank,
S95 router and S98 neural copy scorer. Update only layer_gains on authorized
TRAIN examples routed S93; randomize neither data provenance nor QA splits.
Use balanced six-family replay; include the full answer and EOS in the
objective; add gain sparsity/size penalty to preserve CPU efficiency.
Start with bounded microsteps, saving and rolling back the exact gain and
optimizer state after each proposal.

**C. Controlled ablations and protection:**
Test adding *one* bridge layer at a time, then combinations, to isolate
benefits. Every attempted update must preserve all original 353 full+EOS
correct identities **and** 387 correct first tokens across 456 TRAIN cases
with serialized physical S101 cached CPU runtime, including S95/S98 routes.
No aggregate accuracy-only acceptance.

**D. Promotion gate:**
Require >=1 genuinely new correct full+EOS TRAIN answer, 0 lost previously
correct answers or first tokens, inclusive <=0.700000 bpp, measured cached
CPU speed >=90% of S82 reference on the same host, and CPU/GPU parity.
Once frozen, a newly preregistered never-inspected QA comparison is required
to substantiate generalization; past exposed S100 42/64 QA cannot be reused
for optimization, model selection, or a fresh independent claim.

If the CPU overhead is too high or new gains regress, roll back. Do not
assume that hooking every layer automatically improves quality: a
single reused low-rank mapping can be inappropriate for earlier/later
hidden-state distributions.

## Boundaries

Do not merge to main; do not change Jessica LIVE, Render, Supabase, APK or
any existing accepted research artifact. Do not run tests in Jessica's chat.
This branch holds only a proposed architecture for separate GitHub-based
research validation. Never claim that zero gain connections are already
a trained improvement.
