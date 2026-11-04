"""Recompute true lexical target access and REAL answers in complete 135M CB59."""
import argparse,hashlib,json,re
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v118.archive import Paid
from v118.runtime import attach,restore
from paid_lowrank_v76 import make_model
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import CASES,TOP_K
from .lexical import calculate

def normalize(s):return re.sub(r"[^\w]+"," ",s).strip().casefold()
def evaluate(model,tok):
    observations=[]
    for idx,(q,truth) in enumerate(CASES):
        prompt=build_chat(tok,q)
        src=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        probes=calculate(model,tok,src,truth)
        with torch.inference_mode():
            gen=generate_one(model,tok,prompt,stop_at_eos=True)
        observations.append({"question":q,"gold_for_DIAGNOSTIC_ONLY":truth,
                             "corrected_tokenization":probes,
                             "genuine_autoregressive_response":gen["text"][:260],
                             "real_answer_matches_true_text":normalize(gen["text"])==normalize(truth),
                             "EOS_present":gen["eos_generated"],
                             "actual_generated_first_four_IDs":gen["ids"][:4]})
    out={"total":len(observations),"whitespace_sensitive_cases":sum(
            r["corrected_tokenization"]["some_gold_encoded_as_leading_whitespace_ONLY_token"]
            for r in observations),
         "real_answer_gold_and_eos":sum(r["real_answer_matches_true_text"] and r["EOS_present"]
                                       for r in observations),
         "rank_tables":{}, "rows":observations}
    for k in TOP_K:
        key=str(k)
        out["rank_tables"][key]={
          "old_INVALID_nonsemantic_topK":sum(r["corrected_tokenization"]["historical_old_topK_inflated"][key]
                                            for r in observations),
          "NEW_unprompted_LEXICAL_token_proxy_topK":sum(
            r["corrected_tokenization"]["lexical_proxy_new_topK"][key] for r in observations),
          "NEW_LEXICAL_after_whitespace_teacher_FORCED_topK":sum(
            r["corrected_tokenization"]["lexical_teacher_forced_prefix_topK"][key]
            for r in observations)}
    return out

def main(file,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    paid=Paid(file.read_bytes())
    if hashlib.sha256(paid.base).hexdigest()!=ORIGINAL_CB59_SHA or EXPECTED_UNIQUE!=TOTAL_UNIQUE:
        raise AssertionError("not original source complete native core")
    physical=out/"frozen_original_135M_CB59.cb59"
    physical.write_bytes(paid.base)
    model,meta,_=make_model("baseline",SimpleNamespace(original=physical,folder=out))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("source original unexpectedly dense")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with torch.inference_mode():
        before=evaluate(model,tok)
        originals=attach(model,paid)
        try:
            after=evaluate(model,tok)
        finally:
            restore(model,originals)
    report={
        "experiment":"v128 critical whitespace leakage correction of old v120 first-token rank",
        "old_first_token_rank_IS_NOT_content_quality":True,
        "full_real_pretrained_model":MODEL,
        "real_original_CB59_checksum_preserved":True,
        "host_core_bpp_EXCLUDING_shared_green":paid.stats()["all_in_bpp"],
        "original_CB59_direct_native":before,
        "v119_full_paid_candidate_direct_native":after,
        "official_independent_v89_TEST_NOT_USED":True,
        "no_ORACLE_gold_used_during_generation":True,
        "no_new_weights_or_training":True,
        "no_green_neural_file_modified":True,
        "main_Render_Supabase_APK_Jessica_LIVE_unchanged":True}
    (out/"v128_lexical_target_audit.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v128_WHITESPACE_BUG_RETEST_COMPLETED",
       "CB59":{k:v for k,v in before.items() if k!="rows"},
       "v119":{k:v for k,v in after.items() if k!="rows"},
       "bpp":paid.stats()["all_in_bpp"]},ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("host",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();main(a.host,a.out)
