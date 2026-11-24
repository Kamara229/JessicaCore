"""Z91 matched diversity versus repetition preregistration.
Both arms same 98 historical TRAIN examples, 48 new answer exposures,
identical target tokens and order; only new prompt contexts differ.
No physical training or held-out scoring here.
"""
import json
from collections import Counter
from research.calibri_nra3 import s119q2i8z87_short_word_curriculum as z87
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
from research.calibri_nra3 import s119q2i8z55_gpt_curriculum as z55
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
ARMS=("REPEATED_DEFINITION","DIVERSE_CONTEXTS")
EPOCHS=4
def prepare():
    new=z87.fresh()
    old=z59.curriculum()
    definitions={r["completion"]:r for r in new if r["family"]=="definition"}
    assert len(definitions)==12
    result={}
    for arm in ARMS:
        additions=[]
        for r in new:
            v=dict(r)
            if arm==ARMS[0]:
                v["prompt"]=definitions[r["completion"]]["prompt"]
                v["kind"]=definitions[r["completion"]]["kind"]
                v["family"]="definition"
            additions.append(v)
        rows=[]
        for i,r in enumerate(additions):
            rows+=old[2*i:2*i+2]+[r]
        rows+=old[96:]
        assert len(rows)==146 and len({r["id"] for r in rows})==146
        assert all(r["split"]=="TRAIN" and r["origin"]==z55.ORIGIN for r in rows)
        result[arm]=rows
    a,b=[result[x] for x in ARMS]
    assert [(r["id"],r["completion"]) for r in a]==[(r["id"],r["completion"]) for r in b]
    assert sum(r["prompt"]!=s["prompt"] for r,s in zip(a,b))==36
    assert all(r==s for r,s in zip(a,b) if not r["id"].startswith("S119Q2I8Z87_"))
    return result
def verify(tok=None):
    data=prepare()
    out={"stage":"Z91_MATCHED_CONTEXT_PREREG","arms":ARMS,"epochs_per_arm":EPOCHS,
         "replay_per_epoch":98,"new_exposures_per_epoch":48,"rows_per_epoch":146,
         "unique_new_prompts":{ARMS[0]:12,ARMS[1]:48},
         "fixed_parent":"revived Z73","same_target_order":True,"only_context_diversity_differs":True,
         "planned_updates_total":1168,"planned_cached_target_exposures_total":14576,
         "manifest_SHA256":{a:z55.digest(r) for a,r in data.items()},
         "no_FP32":True,"no_DEV_scoring":True,"no_model_forwards":True,
         "physical_training_requires_prior_Z88_result":True,"candidate_accepted":False,"output_gate":0}
    if tok is not None:
        encoded={a:[verified_native_answer(tok,r) for r in rs] for a,rs in data.items()}
        a,b=[encoded[x] for x in ARMS]
        assert [(r["id"],r["answer_native_with_eos"]) for r in a]==[(r["id"],r["answer_native_with_eos"]) for r in b]
        assert sum(len(r["answer_native_with_eos"]) for r in a)==1822
        out["exact_native_target_parity"]=True
        out["native_positions_per_arm_per_epoch"]=1822
        out["native_plan_SHA256"]={arm:z55.digest([{"id":r["id"],"prefix":r["prefix"],"targets":r["answer_native_with_eos"]} for r in rs]) for arm,rs in encoded.items()}
    return out
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    print("Z91_PREREG "+json.dumps(verify(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
