"""Z80 bind the successful Z70 retry, not the failed preregistered run.
No changes to Z72 questions, scoring or arm definitions.
"""
from research.calibri_nra3.s119q2i8z72_head_eval_prereg import ARMS,verify,require
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import CHILD_SHA,GT_SHA
DEV_SHA="af59cffcb048ef2a95a3877c5fd2e59edbe6afdc187165fce18d68f9348382f0"
PINS={"run_id":45465959517,"head_sha":"eff1fd50ecee21786d8bfb33060b842538464031","artifact_id":14956010931,"conclusion":"success",
      "joint_sha256":{"HEAD_UTF8_BASE":"e93800e6a6196e3853a8928fd8826571c4bb52a9717201a8c11ec87a15c85212",
      "HEAD_UTF8_PENDING_X4":"80462b5bff4d4a7958b0321482165560fa0c587786627b1aafb29646ffc7191e"}}
def verify_pins(report):
    require(verify()["sha256"]==DEV_SHA,"unchanged exact Z72 precommitted answers")
    require(report["stage"]=="S119Q2I8Z70_GPT_JOINT98_PAIR_EOS_UTF8" and report["Z55_joint_SHA256"]==CHILD_SHA and report["Z55_GT12_SHA256"]==GT_SHA
        and report["total_cached_training_positions"]==25888 and report["total_Adam_updates"]==1568
        and report["total_actual_decoder_forwards"]==3302 and report["FP32_calls"]==0 and report["no_DEV_scoring"]
        and report["output_gate"]==0 and not report["candidate_accepted"],"both successful actual cached arms")
    require(report["cache_proof"]["all_positions_top1_equal"] and report["cache_proof"]["real_compressed_decoder_forwards"]==1618,"cached parity")
    require(set(report["saved_research_only"])==set(report["training"])==set(PINS["joint_sha256"])==set(ARMS[1:3]),"both arms")
    for a in ARMS[1:3]:
        r=report["training"][a];m=report["saved_research_only"][a]
        require(r["Adam_updates"]==784 and r["cached_training_positions"]==12944 and r["trainable_parameters"]==247040
            and r["pending_UTF8_multiplier"]==(4 if a==ARMS[2] else 1),"matched original training")
        require(m["joint"]["sha256"]==PINS["joint_sha256"][a] and m["GT12"]["sha256"]==GT_SHA,"exact actual saved checkpoint and frozen GT12")
