"""Z101 exact Z99 preregistration and completed physical Z97 provenance."""
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import PARENT_SHA,GT_SHA,NATIVE_SHA,FRESH_SHA
DEV_SHA="45de9630fb5a341537268747e08750e58d2256eb944f03899008d5469af3b096"
DEV_NATIVE_SHA="2b3687949f26d6c38e40b72a6f62c248502b4c82990c91b3c1ee2c6148625a80"
Z97_SHA="77854e11ff386aacde5eb80a64a025d39663a474ba6695ddb717e8246c385162"
PINS={"Z96":{"run_id":45477612045,"artifact_id":14960046982,"joint_sha256":"292edde21c3128bdddb2e510c29db0f7fed452621ae9504f2bebf38bea83f917"},
      "Z97":{"run_id":45478264277,"artifact_id":14961393358,"joint_sha256":Z97_SHA}}
def verify_z97(r):
    t=r["training"];m=r["saved_research_only"]
    check(r["stage"]=="S119Q2I8Z97_GPT_BATCH_WORD_REPLAY" and r["Z73_revived_parent_SHA256"]==PARENT_SHA
          and r["GT12_SHA256"]==GT_SHA and r["native_plan_SHA256"]==NATIVE_SHA and r["new_curriculum_SHA256"]==FRESH_SHA
          and t["Adam_updates"]==588 and t["total_cached_training_positions"]==187908 and t["replay_group_visits"]==[42]*14
          and r["cache_proof"]["real_compressed_decoder_forwards"]==1822 and r["cache_proof"]["all_positions_top1_equal"]
          and r["short_word_metrics"]["AFTER"]["correct"]==3 and r["short_word_metrics"]["AFTER"]["n"]==48
          and r["TRAIN42_post_audit"]["AFTER"]["all"]["tokens"]==842 and r["total_actual_decoder_forwards"]==4379
          and m["joint"]["sha256"]==Z97_SHA and m["GT12"]["sha256"]==GT_SHA
          and r["FP32_calls"]==0 and r["no_DEV_scoring"] and r["output_gate"]==0 and r["candidate_accepted"] is False
          and not r["inference_masking"],"actual complete Z97 research-only pinned artifact")
