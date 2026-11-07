"""v233 read-only counterfactual label diagnostic: does 'red' encode No?
Identical paired FP32/packed v228 prompts; fixed weights; no training/selection.
"""
from collections import Counter, defaultdict
import gc, hashlib, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from v196 import protocol as P
from v196.recover import build_trainable_from_paid_source
from v229.run import sha_checked, ACCEPTED_V228_STEP12_SHA256
from v176.operators import clear
from v232.run import freeze_input, generate, normalize

# Explicitly defined BEFORE loading any neural weights.
# Both permutations are essential: red cannot be assumed to mean No.
FORMATS = {
 "literal": ("Answer with exactly one word: Yes or No.", {"yes":"yes","no":"no"}),
 "red_yes": ("Answer with exactly one word: red if Yes, blue if No.", {"yes":"red","no":"blue"}),
 "red_no": ("Answer with exactly one word: blue if Yes, red if No.", {"yes":"blue","no":"red"}),
 "true_false": ("Answer with exactly one word: True if Yes, False if No.", {"yes":"true","no":"false"}),
}

def build(tok):
    cases,_=freeze_input(tok)
    # Independent VALID: six binding pairs and six numeric sides; includes
    # positive/negative examples. Existing public BoolQ: frozen cases only.
    valid=[c for c in cases if c["task"] in ("VALID_yesno_binding","VALID_yesno_numeric")]
    assert len(valid)==12
    boolq=[c for c in cases if c["task"]=="boolq"]
    chosen=[]
    for label in ("Yes","No"):
        chosen += [c for c in boolq if c["gold"]==label][:12]
    assert len(chosen)==24
    rows=[]
    for c in valid+chosen:
        # Reconstruct only the factual question, never insert its gold label.
        if c["task"]=="boolq":
            # Frozen prompt contains passage and question; strip old instruction.
            content=c["prompt"].split("[/INST]")[0] if False else None
            # Prefer extracting original user content from tokenizer's chat framing
            # via exact substring, preserving the original passage.
            from v232.run import MAX_CONTEXT_TOKENS
            raw=c["prompt"]
            marker="Answer with the word Yes or No, without any explanation."
            assert marker in raw
            body=raw.split(marker)[0]
        else:
            raw=c["prompt"]
            # VALID templates contain their own explicit answer instruction.
            # Replace only the final instruction, not the factual proposition.
            body=raw.replace("Reply with only Yes or No.", "")
            assert body!=raw
        for fmt,(instruction,mapping) in FORMATS.items():
            if c["task"]=="boolq":
                prompt=body+instruction+raw.split(marker)[1]
            else:
                prompt=body+instruction
            ids=tok.encode(prompt,add_special_tokens=False)
            assert 1<len(ids)<=MAX_CONTEXT_TOKENS
            rows.append({"task":c["task"],"case_id":c["case_id"]+"__"+fmt,
              "source_case_id":c["case_id"],"format":fmt,
              "question":c["question"],"gold":mapping[normalize(c["gold"])],
              "original_gold":c["gold"],"prompt":prompt,"input_ids":ids,
              "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()})
    assert len(rows)==144
    return rows

def evaluate(name,model,sources,tok,rows,out):
    path=out/(name+"_v233.jsonl")
    with path.open("w") as f:
        for i,c in enumerate(rows):
            x=generate(model,sources,tok,c)
            x.update(format=c["format"],source_case_id=c["source_case_id"],
                     original_gold=c["original_gold"],mapped_gold=c["gold"])
            f.write(json.dumps(x,ensure_ascii=False)+"\n")
            f.flush()
            if (i+1)%24==0:print(name,i+1,"/",len(rows),flush=True)

def run(src,out):
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=build(tok)
    (out/"PRECOMMITTED_v233_manifest.json").write_text(json.dumps([
      {k:v for k,v in c.items() if k!="input_ids"} for c in rows],indent=2))
    sealed=sha_checked(src,ACCEPTED_V228_STEP12_SHA256)
    fp=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    assert sum(p.numel() for p in fp.parameters())==EXPECTED_UNIQUE
    evaluate("FP32",fp,None,tok,rows,out)
    del fp;gc.collect()
    packed,paid=build_trainable_from_paid_source(sealed)
    packed.cpu().eval()
    assert len(paid)==P.CODEBOOK_MATRIX_COUNT
    evaluate("CALIBRI_v228",packed,paid,tok,rows,out)
    del packed,paid;gc.collect()
    a=[json.loads(s) for s in (out/"FP32_v233.jsonl").read_text().splitlines()]
    b=[json.loads(s) for s in (out/"CALIBRI_v228_v233.jsonl").read_text().splitlines()]
    assert len(a)==len(b)==len(rows)
    report={"n_per_model":len(rows),"unchanged_physical_bpp":0.6678394131307638,
      "model_sha":MODEL_SHA,"sealed_sha256":ACCEPTED_V228_STEP12_SHA256,
      "no_QA_training_or_weight_selection":True,"results":{}}
    for name,arr in (("FP32",a),("CALIBRI_v228",b)):
        report["results"][name]={}
        for fmt in FORMATS:
            subset=[x for x in arr if x["format"]==fmt]
            report["results"][name][fmt]={
                "n":len(subset),"strict_correct":sum(x["strict_correct"] for x in subset),
                "output_histogram":Counter(x["normalized_actual"] for x in subset).most_common(),
                "by_expected":{v:{"n":sum(normalize(x["expected"])==v for x in subset),
                   "correct":sum(x["strict_correct"] for x in subset if normalize(x["expected"])==v)}
                   for v in set(normalize(x["expected"]) for x in subset)}}
    (out/"ACTUAL_v233_red_no_counterfactual.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"stage":"v233_complete","results":report["results"]}),flush=True)

if __name__=="__main__":
    import sys
    run(sys.argv[1],sys.argv[2])
