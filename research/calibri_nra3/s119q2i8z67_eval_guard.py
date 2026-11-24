"""Fail closed unless both Z65 arms completed, explicitly pinned by SHA.
No checkpoint selection from metrics. Z66 questions scored only once.
"""
import re
from research.calibri_nra3.s119q2i8z66_gpt_paired_eval_prereg import verify,rows,summarize,ARMS
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import CHILD_SHA,GT_SHA
DEV_SHA="ca6c790d795e360890a31cca4a68adee4f6f9a84a460774fe0c84d4f5e04bcbe"
TRAIN_ARMS=("FULL_CE_PLUS_GPT_PAIR_EOS","FULL_CE_PLUS_GPT_PAIR_EOS_UTF8")
def check(ok,msg):
    if not ok:raise ValueError(msg)
def verify_pins(report,pins):
    check(verify()["sha256"]==DEV_SHA,"immutable Z66 prompts")
    check(pins["run_id"]==45435892186 and pins["conclusion"]=="success" and
          pins["head_sha"]=="b80cb13b4d056a716be3fb5f55c34a2cd82836ff",
          "explicit successful Z65 run pin required")
    check(report["stage"]=="S119Q2I8Z65_GPT_JOINT98_PAIR_EOS_UTF8"
          and report["Z55_joint_SHA256"]==CHILD_SHA and report["Z55_GT12_SHA256"]==GT_SHA
          and report["FP32_calls"]==0 and report["no_DEV_scoring"]
          and report["total_actual_training_forwards"]==3236
          and report["total_actual_replay_forwards"]==1684
          and report["total_Adam_updates"]==196
          and report["output_gate"]==0 and not report["candidate_accepted"],
          "both actual matched training arms completed")
    check(set(pins["weights"])==set(report["saved_research_only"])==set(report["training"])==set(TRAIN_ARMS),
          "all fixed arms required, no cherry picking")
    for arm in TRAIN_ARMS:
        train=report["training"][arm];saved=report["saved_research_only"][arm]
        check(train["Adam_updates"]==98 and train["native_training_forwards"]==1618
              and train["GT12_learning_rate"]==.0015 and max(train["GT12_abs_max_history"])<.5
              and train["FP32_calls"]==0 and train["UTF8_loss_positions"]==(1618 if arm==TRAIN_ARMS[1] else 0),
              "same physical training exposure, stability, only UTF8 difference")
        for part in ("joint","GT12"):
            sha=pins["weights"][arm][part]
            check(bool(re.fullmatch("[0-9a-f]{64}",sha)) and saved[part]["sha256"]==sha,
                  "explicit pinned checkpoint SHA")
def results(records):
    check(set(records)==set(ARMS),"exact precommitted arms")
    summary={a:summarize(records[a]) for a in ARMS}
    a,b=summary[ARMS[1]],summary[ARMS[2]]
    gate={"min_UTF8_count_gain":b["valid_UTF8"]-a["valid_UTF8"]>=4,
          "no_complete_answer_regression":b["complete_correct"]>=a["complete_correct"],
          "no_paired_complete_regression":b["both_members_complete_correct"]>=a["both_members_complete_correct"]}
    return {"metrics":summary,"exploratory_gate":gate,"all_exploratory_gates_pass":all(gate.values()),
            "FP32_comparison_only":True,"candidate_accepted":False,"output_gate":0}
