# V005 preflight, pending V004 gate

The historic NRA3 intervention patches only the last token of each prefix.
A naive teacher-forced multi-position pass changes late-layer K/V of earlier
positions and may alter later logits. Causal slicing alone cannot guarantee
equivalence. Compare actual native prefix runs (including slot 2) against:
1. Full teacher-forced sequence, injection only at each respective target in
   separate diagnostic runs: prefix causality control, should match baseline.
2. All target positions injected in a single traversal: measure later-position
   deviation, exact top1/EOS and selected gradient difference.
3. Change suffix tokens with the target prefix held constant: verify no future
   token leakage into earlier predictions.
Keep V001 compact, V002 source freeze, V003 training cache and validated V004
batch path. Equal target/loss normalization. If multi-position math differs,
do not present it as a pure speed optimization or promote it on a tiny probe.
Consider exact sharing of early prefix states with independent late branches,
or a separately trained architecture candidate with independent quality gates.
No weight update in this preflight. Do not schedule until V004 is validated.
