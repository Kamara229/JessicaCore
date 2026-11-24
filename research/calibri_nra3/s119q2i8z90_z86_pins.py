"""Exact physical Z86 artifacts; no model selection from TRAIN scores."""
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z85_balanced_margin import ARMS,PARENT_SHA,GT_SHA
PINS={"REVIVED_BASE":"4a7b31f84c283cc16200f6246e4d79c2117dbbc5cbdd3ebba2ae9d3d0d2edad0",
"REVIVED_BALANCED_STATE_MARGIN":"8ba2e1c4f22eb112a7f61b0612702809ba364299fb3cf671404fa49cf97c08c5"}
def verify_z86(report):
    check(report["stage"]=="S119Q2I8Z86_GPT_JOINT98_PAIR_EOS_UTF8"
          and report["Z73_revived_parent_SHA256"]==PARENT_SHA and report["Z55_GT12_SHA256"]==GT_SHA
          and report["total_cached_training_positions"]==12944 and report["total_actual_decoder_forwards"]==3302
          and report["total_actual_replay_forwards"]==1684 and report["total_Adam_updates"]==784
          and report["FP32_calls"]==0 and report["no_DEV_scoring"] and report["output_gate"]==0
          and report["candidate_accepted"] is False and report["cache_proof"]["all_positions_top1_equal"],
          "completed matched Z86 physical training")
    check(set(report["training"])==set(report["saved_research_only"])==set(ARMS),"both actual arms")
    for a in ARMS:
        m=report["saved_research_only"][a];t=report["training"][a]
        check(m["joint"]["sha256"]==PINS[a] and m["GT12"]["sha256"]==GT_SHA
              and t["Adam_updates"]==392 and t["cached_training_positions"]==6472
              and t["parent_joint_SHA256"]==PARENT_SHA,"pinned actual complete FP16 checkpoint")
