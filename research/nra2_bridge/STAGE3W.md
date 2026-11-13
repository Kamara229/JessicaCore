# NRA2 S3W: isolate why novel grammar failed without touching weights

Original S3V (#43529457952) completed **SUCCESS** as an auditable experiment, **not** as a successful quality result: on 64 S3T new-grammar prompts, soft 33/64, hard 34/64, exact graph 64/64, source 34/64, target 34/64, oracle 64/64; original 616 J7 protected by S3U.

S3T reversed the semantic source/destination **mention order** in question styles 0 and 3 (32 items) versus 1 and 2 (32 items). This stage reads only already produced immutable S3V results and S3T prereg, validates all 64 questions and all existing error records, then slices results by four original templates and by mention order. No changes to adapter, NRA, grammar, thresholds or training, no blind claim or acceptance. Do not test ordinary Jessica or change LIVE, Render, Supabase, QA/TEST.
