"""Z81 exact completed Z75 physical parents and fixed TRAIN42 diagnosis."""
from research.calibri_nra3.s119q2i8z75_joint_after_head import ARMS,PINS as PARENT_PINS
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import CHILD_SHA,GT_SHA
from research.calibri_nra3.s119a_hard_four_goals import check
PINS={"HEAD_CONTROL":{"joint":"6b55ac2923fa4dce665827f14ff2f7fb90159e68e97e11681cd979374a75177b","GT12":"b5555570d204ac8fbb54c8b454e5c1fa20dac538387ab449e180269a38cef45b"},
"HEAD_REVIVE_ZERO_UNITS":{"joint":"dc5ec56a53ce32f920d44431ffde1a4996a5081dd8756b407148656c21758dd4","GT12":"4fa38f072c758cc3c1343634469dea2f52ae9e837592c1455d4671df2339ac9f"}}
def verify_z75(report):
    check(report["stage"]=="S119Q2I8Z75_GPT_JOINT98_PAIR_EOS_UTF8" and report["Z73_joint_pins"]==PARENT_PINS
          and report["Z55_joint_SHA256"]==CHILD_SHA and report["Z55_GT12_SHA256"]==GT_SHA
          and report["total_actual_training_forwards"]==3236 and report["total_actual_replay_forwards"]==1684
          and report["total_Adam_updates"]==196 and report["FP32_calls"]==0 and report["no_DEV_scoring"]
          and report["output_gate"]==0 and report["candidate_accepted"] is False,"successful actual Z75 joint training only")
    check(set(report["training"])==set(report["saved_research_only"])==set(ARMS),"both matched arms")
    for a in ARMS:
        t=report["training"][a];m=report["saved_research_only"][a]
        check(t["Adam_updates"]==98 and t["native_training_forwards"]==1618 and t["starting_child_SHA256"]==PARENT_PINS[a]
              and t["trainable_parameters"]==368051 and max(t["GT12_abs_max_history"])<.5
              and all(m[k]["sha256"]==PINS[a][k] for k in ("joint","GT12")),"exact completed physical child and sidecar")
