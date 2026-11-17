# Oscar v1L — identify whether source-codebook or which Green Tunnel causes v1K regression

v1K did 72 real original 42M independently pretrained FP32 teacher updates of 4 existing K32/V8 source codebooks/row scales PLUS v1F FIRST & v1G LATER 8-layer recurrent GTs, physically repacked and saved. But external FP32 full 8-token parity on 36 PUBLIC debug remained **0/36**, first accuracy **19/36→11/36**, mean last decoder cosine **0.465→0.424**, unigram repetition **0.09→0.35**, despite better TRAIN KL/hidden loss. The main question is **which joint component drives this regression**. v1L must use ONLY existing physically trained v1H/v1K files — **no gradient training** and NO unpublished answer labels for choosing routes. ALL comparisons are exploratory (previously viewed DEBUG), not clean heldout.

### Physically authenticated existing sources

- Original independently pretrained Karpathy TinyStories FP32: 41,689,600 unique 8×512 weights, checkpoint SHA `fbe124df8794c2e033432c22b59cff64485455b4644046c52a456d0bbd8859fe`; original same-geometry model, **NOT CALIBRI 12×320**.
- Original 5-bit core CB59 SHA `f7b5fe62edba2edfe9ef98b79536685f7d8182e9c5cd19e0b092ab15fc68f5f1`, CB61 SHA `724ac4086910859487772a78d7a264047ca1543ffa97af867477aec7857c372a`.
- Original v1F FIRST original rank32 full8 GT FP16 SHA `2fc4b5c989257b70a1d76a20c68eacbcf9be079d0add2912e684969926933543`, original v1G LATER FP16 SHA `506ea6793069e2b913d1322d506f10743133278a46bdf3bc916b4c08ed1a1d1d`.
- Newly trained v1K actual repacked original CB59 SHA `c03f1a2e798ff0549385bb49d341180b6136d4213bdae5797044e840517592dc`, new CB61 SHA `2d3c3f1388e7c8211b665fa30318b1af3a3531c7fb63a50e462d2de21f7c41ba`, new FIRST GT FP16 SHA `a29f79656e71c5873b200809c7cb9611578ee6754c41a8483986d9a98b7a2016`, new LATER GT FP16 SHA `2fa6e700e9a0fbc92374056d361a6f476c8ed22ee4700441620f22e6cc290ffe`.

### Precommitted factorial experiment / no model training

Run real independent pretrained FP32 versus same original-geometry compressed model on 20 previously viewed v1H PUBLIC debug + 8 previously seen v1I DEBUG + 8 previously seen v1K DEBUG (36 prompts, NONE train). Compare **six** exact original checkpoints:
1. **Baseline**: original 5-bit K32/V8 + old v1F FIRST + old v1G LATER (v1H).
2. **Only source modified**: v1K new four-matrix original 5-bit codebook + old FIRST GT + old LATER GT.
3. **Only GT modified**: original 5-bit core + v1K jointly retrained FIRST and LATER GT.
4. **New source, original FIRST**: new packed source + original FIRST GT + v1K trained LATER GT.
5. **New source, original LATER**: new packed source + v1K trained FIRST GT + original LATER GT.
6. **Joint source and both GT**: v1K physically complete model.

Measure actual external FP32 **first, full 8-token+EOS match**, self-generated repetition and final original teacher hidden cosine on each student's own prefixes, all WITHOUT gradient. Compare published v1H and v1K exact first/complete metrics as source integrity check: v1H **19/36 first,0/36 full**, v1K **11/36 first,0/36 full**. Original source and GT files SHA immutable; no actual hybrid with unpublished FP32 original tensors, all six recombinations are **existing physically PAID 5-bit+2GT** and satisfy ≤0.7 bpp byte budget individually, BUT switching choice was informed by past public debug so do not treat posthoc winner as independently validated candidate. Only output research tables, no new deployed model.

## Scientific interpretation

If source-only preserves earlier first tokens but new FIRST GT degrades them, the GT-first train is main cause; keep first module frozen next. If replacing 4 source matrices without updating GT destroys quality, this is evidence of interlayer co-adaptation. If neither change restores complete FP32 responses, do not conduct additional 12-prompt tuning blindly; re-engineer quantization/group vector assignment and use much larger separate calibration corpus and heldout original FP32 comparisons first. Original CALIBRI S142, NRA2, LIVE/main, Render, Supabase, APK and ordinary Jessica chat unchanged. No original 616 J7, native C 90%, unseen external QA.
