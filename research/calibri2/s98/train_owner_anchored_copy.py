"""S98: TRAIN-only neural second-token span-copy feasibility (not model promotion).

S96's 64 novel prompts were inspected ONCE and NEVER opened here. S97 found
105 wrong-second-token TRAIN answers for S93; the frozen S95 router alone
cannot solve unseen noun copying. This experiment trains a tiny *generic*
owner-anchored positional scorer on the original authorized S82 TRAIN
questions, withholding noun groups and instruction frames for diagnosis.

The runtime scorer sees a prompt and the FIRST WORD ALREADY PRODUCED by
the compressed model; it never sees gold second tokens, S96 examples or any
prompt-answer lookup. It chooses among literal following-word occurrences
after that first word in the input; it cannot invent arbitrary second tokens.

All accuracy is CONDITIONAL on the first word being correct; a later S99
integration must do actual full EOS CPU inference and protect 282 historical
skills. S98 never overrides model output and is never accepted.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82

SEED=980098
DIM=12
WIDTH=8
CAP_BPP=0.7
PROPER_NAME=r"\b[A-Z][a-zA-Z]{3,}\b"


class TinyCopyScorer(nn.Module):
    """Numeric neural weights only, ~113 trainable scalars, no noun table."""
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(DIM,WIDTH),
                     nn.Tanh(),nn.Linear(WIDTH,1))
    def forward(self,features):
        return self.network(features).squeeze(-1)


def requested_owner(text):
    # Select the last repeatedly mentioned proper name from the *prompt*.
    # This is a generic source-side feature, not an answer-string lookup.
    matches=list(re.finditer(PROPER_NAME,text))
    freq=Counter(m.group() for m in matches)
    repeated=[m for m in matches if freq[m.group()]>=2]
    if not repeated:
        return None,matches
    return repeated[-1].group(),matches


def candidates(tok,prompt,first):
    """Return only two-word spans actually present in the prompt.

    The first word is supplied by the already-compressed language model
    at runtime. Tokenize each literal span to verify that the second token
    is copyable in the same 2-token format. No target answer is passed.
    """
    term=tok.decode([int(first)],skip_special_tokens=True).strip()
    if not term or not term.isalpha() or len(term)>60:
        return []
    pat=r"(?<![A-Za-z])"+re.escape(term)+r"\s+([A-Za-z]{2,})(?![A-Za-z])"
    names,entity_mentions=None,None
    owner,entity_mentions=requested_owner(prompt)
    if owner is None:return []
    all_matches=list(re.finditer(pat,prompt,re.IGNORECASE))
    if len(all_matches)<2 or len(all_matches)>12:
        return []
    output=[]
    for rank,m in enumerate(all_matches):
        literal=m.group()
        tokens=tok.encode(literal,add_special_tokens=False)
        if len(tokens)!=2 or tokens[0]!=int(first):
            continue
        start=m.start()
        local=[z for z in entity_mentions if z.end()<=start]
        closest=local[-1] if local else None
        target_prior=[z for z in entity_mentions
                      if z.group()==owner and z.end()<=start]
        dist_owner=min(250,start-target_prior[-1].end()) if target_prior else 250
        dist_entity=min(250,start-closest.end()) if closest else 250
        # All 12 features are *prompt-only*. Exclude the candidate's noun
        # lexical ID, any gold answer and the frozen TRAIN question ID.
        feats=np.asarray([
            1.0,
            dist_owner/250,
            dist_entity/250,
            float(bool(target_prior)),
            float(closest is not None and closest.group()==owner),
            rank/max(1,len(all_matches)-1),
            start/max(1,len(prompt)),
            min(1.0,(len(prompt)-start)/250),
            float(closest is not None and closest.group()!=owner),
            float(bool(target_prior) and start-target_prior[-1].end()<=45),
            float(bool(target_prior) and start-target_prior[-1].end()<=90),
            min(1.0,len(m.group(1))/16),
        ],dtype=np.float32)
        output.append((tokens[1],feats))
    return output


def frozen_train_pairs(tok):
    pairs,_=s82.newly_authored_train(tok)
    if len(pairs)!=128:
        raise AssertionError("Changed immutable TRAIN S82 noun pairs")
    samples=[]
    for p in pairs:
        n=int(p["pair_id"])-82000
        if n<0 or n>=128:
            raise AssertionError("Unrecognized S82 TRAIN identity")
        noun_group=(n//4)%8
        frame=n%4
        for side in ("left","right"):
            q=p[side]["question"]
            expected=tok.encode(p[side]["answer"],add_special_tokens=False)
            if len(expected)!=2:
                raise AssertionError("S82 TRAIN no longer two-word")
            options=candidates(tok,q,expected[0])
            indices=[i for i,(word,_) in enumerate(options) if word==expected[1]]
            # Distinct owner/noun contrast must permit exactly one literal
            # copy, otherwise report inability instead of false success.
            if len(options)!=2 or len(indices)!=1:
                raise AssertionError(f"No unique literal paired noun copy on TRAIN group={noun_group} frame={frame} side={side}; {len(options)} options, {len(indices)} positives")
            samples.append((noun_group,frame,np.stack([a[1] for a in options]),
                            indices[0],side))
    if len(samples)!=256:
        raise AssertionError("S98 must cover only 256 original S82 TRAIN prompts")
    return samples


def train(feat,goal,indices,seed,steps=700):
    if not torch.backends.mps.is_available():
        raise RuntimeError("S98 requires actual Apple MPS training, no CPU fallback")
    torch.manual_seed(seed)
    model=TinyCopyScorer().to("mps")
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.018,weight_decay=0.0005)
    xx=feat[indices].to("mps")
    yy=goal[indices].to("mps")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=model(xx)
        loss=F.cross_entropy(logits,yy)
        loss.backward()
        optimizer.step()
    model.cpu().eval()
    # Ensure final eval is on exactly physically serialized FP16 values.
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(param.half().float())
    return model


def evaluate(model,feat,goal,indices):
    with torch.inference_mode():
        predicted=model(feat[indices]).argmax(dim=-1)
    return {"correct":int((predicted==goal[indices]).sum().item()),
            "evaluated":len(indices),
            "conditional_second_token_copy_accuracy":
                float((predicted==goal[indices]).float().mean().item())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("S95_frozen_report",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    before=json.loads(a.S95_frozen_report.read_text())
    if (before.get("stage")!="S95_PROMPT_ONLY_NEURAL_THREE_PROFILE_ROUTER_TRAIN_RESEARCH"
        or not before.get("all_456_EOS_output_exact_routed")
        or before.get("candidate_accepted") is not False
        or not before.get("physical_bpp_at_most_0_7")):
        raise AssertionError("S98 requires frozen S95 actual CPU, not a new base")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    data=frozen_train_pairs(tok)
    feat=torch.as_tensor(np.stack([row[2] for row in data]),dtype=torch.float32)
    goal=torch.as_tensor([row[3] for row in data],dtype=torch.long)
    train_ids=[i for i,row in enumerate(data) if row[0]<=5]
    held_ids=[i for i,row in enumerate(data) if row[0]>5]
    if (len(train_ids),len(held_ids))!=(192,64):
        raise AssertionError("S98 must hold out exactly two TRAIN noun groups")
    model=train(feat,goal,train_ids,SEED)
    primary=evaluate(model,feat,goal,held_ids)
    trained=evaluate(model,feat,goal,train_ids)
    fvals=[]
    for frame in range(4):
        ids=[i for i,row in enumerate(data) if row[0]<=5]
        fit=[i for i in ids if data[i][1]!=frame]
        leave=[i for i in ids if data[i][1]==frame]
        candidate=train(feat,goal,fit,SEED+frame+1,steps=700)
        fvals.append({"left_out_template_index":frame,
                      **evaluate(candidate,feat,goal,leave)})
    weights={k:v.detach().contiguous().half() for k,v
             in model.state_dict().items()}
    outmodel=a.out/"S98_MPS_TRAINED_NUMERIC_OWNER_COPY_FP16.safetensors"
    save_file(weights,str(outmodel))
    loaded=TinyCopyScorer()
    loaded.load_state_dict(load_file(str(outmodel)),strict=True)
    if evaluate(loaded,feat,goal,held_ids)!=primary:
        raise AssertionError("S98 physical FP16 scorer changed after roundtrip")
    result={
       "stage":"S98_GENERIC_NEURAL_OWNER_ANCHORED_NOUN_COPY_TRAIN_ONLY",
       "frozen_base_S95_456_full_EOS":287,
       "S95_immutable_core_tunnel_router":True,
       "QA_previous_S96_novel_prompts_never_accessed":True,
       "all_256_original_S82_TRAIN_only":True,
       "neural_trained_on_actual_Apple_MPS":True,
       "learned_runtime_contains_no_answers_nouns_ID_or_prompt_lookup":True,
       "runtime_copy_conditioned_on_existing_first_model_word":True,
       "frozen_full_CPU_456_regression_not_yet_rerun":True,
       "actual_full_sequence_EOS_inference_not_yet_tested":True,
       "MPS_tiny_scorer_trainable_parameters":
           sum(p.numel() for p in model.parameters()),
       "physically_serialized_router_extra_bytes":outmodel.stat().st_size,
       "TRAIN_noun_groups_fitted":6,
       "TRAIN_noun_groups_unseen_by_new_scorer":2,
       "TRAIN_fit_conditional_copy":trained,
       "held_out_noun_group_TRAIN_conditional_copy":primary,
       "held_out_prompt_template_TRAIN_folds":fvals,
       "candidate_accepted":False,"LIVE_untouched":True
    }
    note=a.out/"S98_MPS_NEURAL_COPY_FEASIBILITY_REPORT.json"
    # Conservatively double count old S95 report and own new diagnostic
    # bytes. This guarantees no unbilled numerical trainable storage.
    for _ in range(40):
        blob=json.dumps(result,indent=2,sort_keys=True).encode()
        total=before["physical_all_inclusive_bytes_conservative"]+len(blob)+outmodel.stat().st_size
        fields={"manifest_bytes":len(blob),
                "conservative_all_inclusive_physical_bytes":total,
                "conservative_all_inclusive_physical_bpp":round(8*total/SOURCE_PARAMS,12),
                "physical_bpp_guard_passed":8*total/SOURCE_PARAMS<=CAP_BPP}
        if all(result.get(k)==v for k,v in fields.items()):
            break
        result.update(fields)
    note.write_text(json.dumps(result,sort_keys=True,indent=2))
    print(json.dumps({"event":"S98_REAL_MPS_NEURAL_OWNER_NOUN_COPY_FEASIBILITY",
        "trained_second_conditional":trained,
        "new_noun_heldout_TRAIN_second_conditional":primary,
        "leave_one_template_out_TRAIN":fvals,
        "all_inclusive_conservative_bpp":result["conservative_all_inclusive_physical_bpp"],
        "candidate_accepted":False,
        "QA_S96_unseen_never_accessed":True}),flush=True)
    if not result["physical_bpp_guard_passed"]:
        raise SystemExit("S98 rejected: paid numeric scorer exceeds 0.7 bpp")
    # No acceptance of a neural selector without physically generated
    # original TRAIN + genuinely novel unseen QA and speed/EOS guards.


if __name__=="__main__":
    main()
