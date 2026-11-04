"""First-token teacher alignment plus real full decoder output, not semantic proof."""
from statistics import mean
import torch
from torch.nn import functional as F
from v130.scoring import evidence,summarize
from v131.probe import summarize_logits

ARMS=("original_CB59","uniform_recode_CB59","weighted_recode_CB59")


@torch.inference_mode()
def one_case(model,tokenizer,question,other,gold,foil,pair,side,teacher_logits):
    from v131.probe import first_logits
    from v94.generation import generate_one
    from v108.generation import build_chat
    student,_,prompt_len=first_logits(model,tokenizer,question)
    probe=summarize_logits(student,tokenizer,gold,foil,None,prompt_len)
    t=teacher_logits.detach().to(torch.float64)
    z=student.detach().to(torch.float64)
    p=F.softmax(t,dim=-1)
    teacher_kl=float((p*(F.log_softmax(t,dim=-1)-F.log_softmax(z,dim=-1))).sum())
    if teacher_kl< -1e-5:raise ValueError("invalid teacher KL")
    generation=generate_one(model,tokenizer,build_chat(tokenizer,question),stop_at_eos=True)
    return {
        "pair":pair,"side":side,"question":question,
        "opposite_question_FOR_ANALYSIS_ONLY":other,
        "reference_FOR_POST_GENERATION_GRADING_ONLY":gold,
        "foil_FOR_POST_GENERATION_GRADING_ONLY":foil,
        "teacher_to_student_first_token_KL":max(0.0,teacher_kl),
        "first_token_agrees_with_FP32":int(student.argmax())==int(t.argmax()),
        "eos_first":probe["argmax_id"]==probe["eos_id"],
        "gold_first_token_top8_PROXY":min(
            x["rank"] for x in probe["answer_first_token_probe_NOT_full_answer_accuracy"]["expected"])<=8,
        "foil_first_token_top8_PROXY":min(
            x["rank"] for x in probe["answer_first_token_probe_NOT_full_answer_accuracy"]["swapped"])<=8,
        "first_prediction_text":probe["argmax_text"],
        "generated_text":generation["text"],
        "generated_token_ids":generation["ids"],
        "eos_generated":generation["eos_generated"],
        "generated_tokens":len(generation["ids"]),
        "elapsed_seconds":generation["elapsed_seconds"],
        "observation":evidence(generation["text"],question,gold,foil,generation["eos_generated"]),
    }


def summarize_cases(arms):
    if tuple(arms)!=ARMS:raise ValueError("incomplete or reordered experiment arms")
    keys=[(x["pair"],x["side"],x["question"]) for x in arms[ARMS[0]]]
    if len(keys)!=16 or len(set(keys))!=16:
        raise ValueError("not exactly the 16 frozen paired examples")
    out={}
    for name,rows in arms.items():
        if [(x["pair"],x["side"],x["question"]) for x in rows]!=keys:
            raise ValueError("non-matched prompts")
        out[name]={
            "teacher_KL_mean":mean(x["teacher_to_student_first_token_KL"] for x in rows),
            "first_token_exact_teacher_match":sum(x["first_token_agrees_with_FP32"] for x in rows),
            "first_EOS":sum(x["eos_first"] for x in rows),
            "target_token_rank_top8_PROXY":sum(x["gold_first_token_top8_PROXY"] for x in rows),
            "whole_reply_observation":summarize(rows),
            "mean_generated_tokens":mean(x["generated_tokens"] for x in rows),
            "seconds_total":sum(x["elapsed_seconds"] for x in rows),
        }
    out["interpretation_limit"]="16 pre-used prompts; lexical gold visibility is not semantic correctness. Re-encoded embedding is tied to output head, so embedding and head change together; NLL/SciQ/BoolQ not tested."
    return out
