# S3AW: fixed experiment before training on S3AU syntax diversification

Research only. No model acceptance, no LIVE/Render/Supabase/APK/main, no Jessica-chat tests. Gate 0.

**Motivation, pinned evidence.** S3AS #43581666491 succeeds on TRAIN512 (512/512) but DEV128 is 64/128, versus S3Z 37/128. S3AT #43582046510 finds 4 DEV syntax templates: S3AS 31/32, 0/32, 1/32, 32/32. This is a query-language role-generalization deficit, not evidence that the underlying reachability graph logic is correct or that full responses improved.

**Precommitted training set.** S3AU #43582403331 generates 64 already authorized S3AQ TRAIN-only topologies × 8 novel paraphrases = **512 new prompts**; SHA `ac038f50dab0ace449f1df32ec7e8728476420615f60c85a8d8fce64c787558a`. S3AV frozen native 43M CPU capture is in progress as #43582744317. Do not train until it succeeds and the full physical token SHA and 616-J7 audit are pinned.

**Fixed candidate training protocol:**
- Initialize from physically locked S3AS FP16 SHA `06420e22270dd082116086a5174e95c04e9ff2b52664583bc8d5a1b4a074d488`.
- Train only S3O semantic query-role head on original CPU features. Freeze graph branch and normalizer; original 43M CPU, Green Tunnel, independent NRA2 2241p untouched.
- 4 groups: replay S3M TRAIN128, S3X reversal TRAIN128, S3AQ TRAIN256, and novel S3AU TRAIN512. **No S3AQ DEV128 or future EVAL128 examples in optimizer.**
- Fixed seed 2026111495, AdamW lr 0.0001, weight decay 0.02, 160 optimizer steps, norm clipping 1.0, CPU-only. Loss weights (replay 128, reversal 128, S3AQ256, S3AU512) = (2,2,2,1); S3AQ quartet reversal symmetry 0.15; S3AU octet same-role/role-swap consistency 0.15.
- Save FP16 checkpoint, verify physical SHA + original S3O graph bitwise + S3AS parent bitwise, output gate 0. Only **then** run previously scored research DEV128 once on locked S3AS and S3AW, reporting per wording and source/destination order. No tuning or re-training after observing this DEV score in S3AW. Never access future independent EVAL128.
- Require original S194 616 J7 unchanged from native S3AV, no forgetting on existing TRAIN-only rehearsal, and no use of previous scored EVAL as training.
- S3AW remains **not accepted** even if research DEV improves. Full original decoder answers with EOS, CPU speed ≥90%, all-in physical bpp ≤0.7, 616 J7 and independent EVAL proof are separate acceptance gates.
