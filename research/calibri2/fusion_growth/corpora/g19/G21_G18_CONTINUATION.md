# G21 — continuation of G18

G18 held-out exact 0/48, TRAIN subset 0/12; all 16 sentence responses start with token 155, so first-token rank cannot be a comprehension metric. Test the hypothesis that the last early-layer position is too impoverished: learned query/key attention over all early prefix states, initialized to exactly zero contribution. Compare against continued training of the original G18 residual. No permanent router.

Both arms start from artifact 15153025395, residual SHA fb42a068d4c90042bb08d1637b0c1058ebe14a51b9875ffed465827a40e02d4e. G05 weights remain frozen. 192 G19 TRAIN +144 G16 TRAIN replay; DEV 64+48. Twelve matched epochs, all 49152 output logits, EOS training. Real native generation and full first-word metric, malformed UTF-8 replacement, repetition, pair accuracy. No automatic promotion or semantic score inferred from first token.

Prepared only. G16 remains in progress; G20 should be run with complete artifacts before launching another training job. Native tokenization and execution are not yet verified.

```sh
python -m research.calibri2.fusion_growth.g21_g18_context_attention --g05 inputs/g05 --g18 inputs/g18 --f08 inputs/f08 --out results/g21
```
