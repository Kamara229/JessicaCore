"""Conservative extraction of actual generated answers, not teacher logits.

No gold answer is read before inference. Ambiguous prose remains UNGRADED:
do not turn an extraction failure into a claim that FP32 is unable to answer.
"""
import re
from collections import Counter
from .protocol import ARMS,DOMAINS,DOMAIN_COUNTS,FORMATS,EXPECTED_GENERATIONS_PER_ARM

COLORS=("red","blue","green","yellow","purple","orange","black","white")
ONE_TO_TWENTY=("zero","one","two","three","four","five","six","seven","eight",
               "nine","ten","eleven","twelve","thirteen","fourteen","fifteen",
               "sixteen","seventeen","eighteen","nineteen","twenty")

def tokens(text):
    return re.findall(r"\b[a-z0-9]+\b",text.lower())

def detect(candidate):
    txt=candidate["verbatim_text"].strip()
    kind=candidate["kind"]
    words=tokens(txt)
    if not words:
        return "unreadable",None
    if kind=="yesno":
        for start in (r"^\s*(yes|no)\b",r"\banswer\s*(?:is|:)\s*(yes|no)\b"):
            m=re.search(start,txt,re.I)
            if m:return "parsed",m.group(1).capitalize()
        return "ambiguous",None
    if kind=="color":
        present=[v for v in COLORS if v in words]
        if len(present)==1:return "parsed",present[0]
        return ("ambiguous" if len(present)>1 else "unreadable"),None
    if kind=="number":
        m=re.search(r"=\s*(-?\d+)\b",txt)
        if m:return "parsed",str(int(m.group(1)))
        m=re.search(r"\b(?:answer|result|is|equals|are)\s*(?:is|:)?\s*(-?\d+)\b",txt,re.I)
        if m:return "parsed",str(int(m.group(1)))
        nums=re.findall(r"\b\d+\b",txt)
        if len(nums)==1:return "parsed",str(int(nums[0]))
        if not nums:
            word_vals=[str(i) for i,w in enumerate(ONE_TO_TWENTY) if w in words]
            if len(word_vals)==1:return "parsed",word_vals[0]
            return "unreadable",None
        return "ambiguous",None
    if kind in ("name","word"):
        # Absence of reference is a certain miss; presence can be a mere
        # restatement, so avoid claiming semantic correctness for >1 names.
        gold=candidate["gold"].casefold()
        hits=re.findall(r"\b"+re.escape(gold)+r"\b",txt.casefold())
        if hits:return "candidate_gold_present",candidate["gold"]
        return "unreadable",None
    raise ValueError("invalid explicit answer kind: "+kind)

def grade(case,answer):
    merged={**case,**answer}
    tag,extracted=detect(merged)
    gold=str(case["gold"]).lower().strip()
    if extracted is None:
        correct=None
    elif tag=="candidate_gold_present":
        # gold-string is only a positive lexical clue, not a semantic pass.
        correct=None
    else:
        correct=str(extracted).lower().strip()==gold
    return {"extraction_status":tag,"extracted_answer":extracted,
            "verified_format_correct":correct,
            "gold_lexical_presence_only":extracted==case["gold"] and tag=="candidate_gold_present"}

def summarize(cases,results):
    if tuple(results)!=ARMS or len(cases)!=EXPECTED_GENERATIONS_PER_ARM:
        raise ValueError("missing model or exact number of frozen prompts")
    out={}
    all_arms={}
    for arm in ARMS:
        records=results[arm]
        if len(records)!=len(cases):
            raise ValueError(f"candidate {arm} skipped a question")
        joined=[]
        for c,r in zip(cases,records):
            if c["prompt_sha"]!=r["prompt_sha"] or c["question_id"]!=r["question_id"] or c["style"]!=r["style"]:
                raise ValueError("mismatched questions across original models")
            verdict=grade(c,r)
            joined.append({**r,**verdict,"gold":c["gold"],"kind":c["kind"],
                           "pair_id":c["pair_id"]})
        all_arms[arm]=joined
        summary={}
        for dom in DOMAINS:
            for style in FORMATS:
                selection=[x for x in joined if x["domain"]==dom and x["style"]==style]
                n=DOMAIN_COUNTS[dom]
                if len(selection)!=n:
                    raise ValueError("missing domain/style text responses")
                verified=[x for x in selection if x["verified_format_correct"] is not None]
                summary[f"{dom}:{style}"]={
                    "all_answers":len(selection),
                    "machine_parsed":len(verified),
                    "machine_verified_correct":sum(x["verified_format_correct"] for x in verified),
                    "ambiguities_unscored":len(selection)-len(verified),
                    "response_token_count":sum(x["output_tokens"] for x in selection),
                    "mean_token_distinctness":sum(x["distinct_token_fraction"] for x in selection)/n,
                    "extraction_status_histogram":dict(Counter(x["extraction_status"] for x in selection)),
                }
        total_tokens=sum(x["output_tokens"] for x in joined)
        total_seconds=sum(x["full_wall_seconds"] for x in joined)
        if not total_seconds>0 or not total_tokens>0:
            raise ValueError("invalid real native token timing")
        out[arm]={
            "by_domain_and_prompt_style":summary,
            "total_output_tokens":total_tokens,
            "full_model_seconds":total_seconds,
            "observed_tokens_per_second":total_tokens/total_seconds,
            "responded_to_every_question":len(joined)==EXPECTED_GENERATIONS_PER_ARM,
        }
    out["CB161"]["measured_speed_vs_original_CB59"]=(
        out["CB161"]["observed_tokens_per_second"]/
        out["CB59"]["observed_tokens_per_second"])
    return {
        "summary":out,"case_records":all_arms,
        "not_a_human_semantic_grade":True,
        "ambiguous_text_never_auto_marked_wrong":True,
        "one_word_reference_in_long_answer_is_not_sufficient_to_claim_correct":True,
        "no_KL_NLL_PPL_or_teacher_logit_metric":True,
    }
