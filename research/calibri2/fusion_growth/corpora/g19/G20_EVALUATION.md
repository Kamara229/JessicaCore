# G20 — fixed Russian cross-evaluation

Run only after complete G16, G17 and G18 artifacts are available and verified. No new workflow is added while these three jobs run.

Six predeclared arms: G05, G16 epochs 1 and 2, G17 raw and centered, G18 early. Each receives 128 unchanged G14 questions and 64 untouched G19 DEV questions. No training on either test.

Metrics: exact normalized answers and EOS, diversity, both correct and completed in a contrast pair. A changed response alone is not correctness. Raw generated answers and a shuffled blinded reviewer CSV are exported, with separate identity key. Human semantic/grammar/grounding scores remain blank until actually reviewed. Preserve all failed answers.

Use the pinned F08 tokenizer and G05, completed G16/G17/G18 artifact directories:

```sh
python -m research.calibri2.fusion_growth.g20_cross_evaluation --g05 inputs/g05 --g16 inputs/g16 --g17 inputs/g17 --g18 inputs/g18 --f08 inputs/f08 --out results/g20
```

Known physical payload bytes include tokenizer and residual. This is not final CPU, historical 616, or all-in runtime certification. Do not promote a checkpoint automatically. G14 has been viewed previously; G19 is held-out but template-based. Code prepared, not executed before candidate artifacts exist.
