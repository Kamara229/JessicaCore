"""Z75 matched physical full-joint continuation of both Z73 heads."""
from research.calibri_nra3.s119q2i8z65_stable_gate_train import train as joint_train,ARMS as JOINT_ARMS,prepare,PLAN_SHA,CURRICULUM_SHA,STATE_SHA
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import CHILD_SHA,GT_SHA
ARMS=("HEAD_CONTROL","HEAD_REVIVE_ZERO_UNITS")
PINS={"HEAD_CONTROL":"637cee708fa33d012bc619006b2b1fd0671fb7eb565ec1b9914e4d3446c80c7b",
"HEAD_REVIVE_ZERO_UNITS":"25acb994cfb8f6e425127da6419b0f47fa5c955b6bbf69d888186e985b1c5256"}
def verify_z73(report):
    check(report["stage"]=="S119Q2I8Z73_GPT_JOINT98_PAIR_EOS_UTF8"
          and report["Z55_joint_SHA256"]==CHILD_SHA and report["Z55_GT12_SHA256"]==GT_SHA
          and report["total_cached_training_positions"]==12944 and report["total_actual_decoder_forwards"]==3302
          and report["total_Adam_updates"]==784 and report["FP32_calls"]==0 and report["no_DEV_scoring"]
          and report["output_gate"]==0 and report["candidate_accepted"] is False
          and set(report["training"])==set(report["saved_research_only"])==set(ARMS),
          "only both completed fixed Z73 arms")
    for a in ARMS:
        t=report["training"][a];m=report["saved_research_only"][a]
        check(t["Adam_updates"]==392 and t["cached_training_positions"]==6472 and t["FP32_source_calls"]==0
              and t["pending_UTF8_multiplier"]==1 and t["trainable_parameters"]==247040
              and m["joint"]["sha256"]==PINS[a] and m["GT12"]["sha256"]==GT_SHA,
              "same exposure and pinned exact physical parent")
    r=report["training"][ARMS[1]]["revival"]
    check(r["initial_top1_parity"] and r["initial_max_native_logit_difference"]==0 and r["dead_units"]==[1,3],
          "confirmed function-preserving revival")
def train(model,sources,bridge,full,expansion,fit,rows,contrasts,*,arm):
    check(arm in ARMS,"both Z73 arms continued, no selection")
    result=joint_train(model,sources,bridge,full,expansion,fit,rows,contrasts,arm=JOINT_ARMS[1])
    result["arm"]=arm
    result["objective"]="CE_PAIR_EOS_UTF8"
    result["starting_child_SHA256"]=PINS[arm]
    return result
