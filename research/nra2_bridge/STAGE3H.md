# NRA2 S3H — real 43M CPU full-token capture of factorized S3G 128

## Before training

S3F #43514000830 has full-token attention TRAIN32 32/32 but
OOD16 0/16 exact, 7/16 reachability (worse than prior S3D 8/16).
The 16 previous OOD probes were reused and are exploratory only.

This stage reuses frozen S194 43M Calibri+Green Tunnel, the original
616 native generated-token J7 and the frozen 2,241-parameter graph NRA.
It does **NOT** modify or train any model. The S3G preregistered
128 prompts are split 64 TRAIN, 16 matched structure, 16 novel wording,
16 longer graphs, and 16 both shifts. Only the 64 TRAIN rows may be used
for future optimizer steps; the 64 controls must remain unscored here.

Source prerequisite S3E immutable GitHub run **#43512049506**, with
authentic [48,T,320] layer12 token FP16, masks and exact 616 original
CPU output hashes. Preserve physical SHA of that archive in new audit.
Use only prompt text in read-only hooks; **construct all graph labels
after** 616 original generation and 128 additional prefills finish.

Save and physically reload:
- [128,T,320] real last-decoder-layer FP16 tokens, mask and lengths;
- [128,320] last contextual token FP16 states;
- adjacency [128,6,6], source [128,6], target [128,6], reachability;
- exact immutable prompt SHA list per registered split (stored for
  provenance only, never passed to model, adapter, routing or inference);
- original J7 SHA parity, frozen layer and independent graph NRA SHA.

Do not infer model quality from capture CI success. **Next stage S3I**
may fit only the training 64, commit physical FP16 then evaluate
controlled matched/wording/length/both splits *once*, with original
S3F metrics retained as a negative historical baseline. Define
training architecture and all thresholds before opening control labels.

As always gate=0, no change to LIVE Jessica/Render/Supabase/APK/main,
sealed QA/TEST, Calibri/Green Tunnel weights or protected 584/584.
