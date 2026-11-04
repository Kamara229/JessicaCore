# CALIBRI v105: audit of frozen v96 outputs

Source: original v96 Actions #41739479888, artifact #13281385044; independent existing judge-only Actions #41739479897, artifact #13279498295. No weights, retraining, or official TEST opened.

Historical strict-prefix scorer marked all three models 0/8. Direct reading of the first FP32 answers instead supports 4/8 factual answers: half of 18 = 9, Oslo, Pacific Ocean, and ice. FP32 failed the other four arithmetic/factual questions. CB59 and v92 each answered 0/8 and generated nonsensical repetitions on all twelve prompts.

The >75% identical-token alarm detected only 1/12 distorted outputs for each packed model. Unique-token ratio 0.1458 (CB59), 0.1589 (v92), 0.6901 (FP32) does not certify coherence. Original prompts lacked the actual instruct-chat template; FP32 also emitted false role separators. Independent original TRAIN BoolQ in v103 confirms collapsed YES-only native predictions. Conclusion: both packed models fail generative quality; original FP32's 0/8 strict score was itself misleading. Do not promote by CI PASS.

Next: apply original chat template, grade grounded answers, block role-marker leakage, detect repetitive word fragments, and preserve complete decoded texts. Physical candidates must meet <=0.7 bpp and >=90% native CPU speed with meaningful language.
