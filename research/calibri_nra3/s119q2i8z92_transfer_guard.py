"""Z92 source-only preflight. No weight pin can be inferred before Z88 succeeds."""
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import NATIVE_SHA,FRESH_SHA,PARENT_SHA,GT_SHA
DEV_SHA="e12160915556d3de390f5b8e124ca01ffea98fe1633b4fccd0bc9c6ad9ea218a"
DEV_NATIVE_SHA="a6cdf9395ac4e7d70b84d18edc0a2034aa7939c3cc50882c596ef9bd478df9f5"
def verify_z88(report,pins):
    check(pins.get("run_id")==45473255610 and pins.get("head_sha")=="92a1e2bf559e6021989aa592c2eaced5685e79a8"
          and pins.get("conclusion")=="success" and type(pins.get("artifact_id")) is int and pins["artifact_id"]>0,
          "explicit successful actual Z88 artifact pin required")
    h=pins.get("joint_sha256","")
    check(len(h)==64 and all(c in "0123456789abcdef" for c in h) and len(set(h))>1,
          "real explicit checkpoint SHA required; no placeholder")
    check(report["stage"]=="S119Q2I8Z88_GPT_SHARED_WORD_TRAIN"
          and report["Z73_revived_parent_SHA256"]==PARENT_SHA and report["GT12_SHA256"]==GT_SHA
          and report["native_plan_SHA256"]==NATIVE_SHA and report["new_curriculum_SHA256"]==FRESH_SHA
          and report["training"]["Adam_updates"]==584 and report["training"]["cached_training_positions"]==7288
          and len(report["training"]["epochs"])==4 and report["cache_proof"]["real_compressed_decoder_forwards"]==1822
          and report["cache_proof"]["all_positions_top1_equal"]
          and report["short_native_cap"]==16 and report["FP32_calls"]==0 and report["no_DEV_scoring"]
          and report["candidate_accepted"] is False and report["output_gate"]==0 and not report["inference_masking"],
          "completed full Z88 exact training")
    for e in report["training"]["epochs"]:
        check(e["short"]["rows"]==48 and e["short"]["positions"]==204
              and e["replay"]["rows"]==98 and e["replay"]["positions"]==1618,"all fixed exposures")
    for k in ("BEFORE","AFTER"):
        check(report["short_word_metrics"][k]["n"]==48 and report["TRAIN42_post_audit"][k]["all"]["tokens"]==842,
              "complete before and after actual audit")
    check(report["total_actual_decoder_forwards"]==3506+sum(report["short_word_metrics"][k]["forwards"] for k in ("BEFORE","AFTER")),
          "actual decoder count")
    m=report["saved_research_only"]
    check(m["joint"]["sha256"]==h and m["GT12"]["sha256"]==GT_SHA,"exact fitted joint and preserved GT12")
