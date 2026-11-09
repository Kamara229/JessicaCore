"""CALIBRI S82: real MPS GPU universal *contextual token-copy* neural Green Tunnel.

Reason for NEW architecture, not S79 QA fitting:
S78/S80/S81 experiments deal with an existing low-rank late-token
logit correction. This S82 independent hypothesis is a GENERIC
learned pointer from source-token hidden states to next-token hidden:
whether a compressed model can copy task-relevant *arbitrary* words
from its input instead of memorizing every noun in fixed TRAIN. This
requires no text-specific JSON lookup, hand-written answer template at
inference, or lookup by color. Generic token-level neural attention
works over the frozen original 36M model's contextual representations.

Do NOT load/query/score S79, S72, S68, S55, S53, S29 or any inspected QA.
All model choice must depend only on authored TRAIN. No one-shot QA
evaluation in this module, no FP32 quality parity assertion.

Freeze fully physically paid S74 (source 36M CB59/S61, first rank2+16,
later rank128+context gate). Learn only rank16 query/key attention,
scalar gain and scalar per-query gate. Apply only at second generation
step, and never replace an existing top-ranked EOS. Save a candidate
only if full native autoregressive MPS and physically compressed CPU
greedy preserve ALL old S74 full/first correct TRAIN IDs, improve new
S82 authored TRAIN full answers, CPU rate >=0.9 S74, and all bytes
source + controller + manifest <=0.7 bpp. Actual GPU gradients required.

This small-first study is only a trial of a universal token mechanism,
NOT a claim of generality across large models. No LIVE Jessica changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from v176.operators import clear
from v186.pairs import build_pairs,tokenize_one

SEED=820082
RANK=16
STEPS=160
CHECKS=(40,80,120,160)
LR=.002
BASE_S74_FULL_TRAIN=(44,31,7,16,12)
BASE_S74_FIRST_TRAIN=(64,56,7,32,32)
NOUNS=("apple","bread","candle","chair","clock","coat","cup","desk",
       "door","flower","glass","hammer","hat","horse","house","jar",
       "key","lamp","leaf","letter","mirror","pencil","piano","plate",
       "radio","river","scarf","shirt","shoe","spoon","stone","table",
       "tent","ticket","tower","train","tree","vase","window","wheel")
TRAIN_NAMES=(
    ("Folemar","Bintan"),("Nerilo","Kassel"),
    ("Voltari","Meyven"),("Renavi","Calori"),
)
TRAIN_FRAMES=(
    "TRAIN S82 inventory: The record for {a} displays {left}. The separate record for {b} displays {right}. Return the two-word value of {a}.",
    "TRAIN S82 workshop note: Worker {b} has key {right}; worker {a} instead has key {left}. State exactly the assigned two words for {a}.",
    "TRAIN S82 permit index: Under {a} the color-item phrase is {left}, whereas under {b} it is {right}. Copy the complete color and item for {a}.",
    "TRAIN S82 packing label: Package owner {b} received {right}. Package owner {a} received {left}. What is the exact two-word label on {a}'s package?",
)
assert len(TRAIN_FRAMES)==len(TRAIN_NAMES)==4

def sha(blob):return hashlib.sha256(blob).hexdigest()

class GenericContextPointer(nn.Module):
    """Vocabulary-neutral q-key neural pointer; no word table/labels stored."""
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(320,RANK,bias=False)
        self.key=nn.Linear(320,RANK,bias=False)
        self.gate=nn.Linear(320,1,bias=True)
        self.log_gain=nn.Parameter(torch.tensor(3.0))
        with torch.no_grad():
            nn.init.normal_(self.query.weight,std=.02)
            nn.init.normal_(self.key.weight,std=.02)
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    def attention(self,q,keys,mask):
        query=F.normalize(self.query(F.layer_norm(q,(320,))),dim=-1)
        key=F.normalize(self.key(F.layer_norm(keys,(320,))),dim=-1)
        score=torch.einsum("br,bpr->bp",query,key)*7.0
        score=score.masked_fill(~mask,-1.e4)
        attn=torch.softmax(score,dim=-1)
        gain=self.log_gain.exp().clamp(max=50.0)
        gating=torch.sigmoid(self.gate(F.layer_norm(q,(320,)))).squeeze(-1)
        return attn, gain*gating

    def candidate_scores(self,q,keys,mask,association,base):
        attn,gain=self.attention(q,keys,mask)
        weights=torch.einsum("bp,bpc->bc",attn,association)
        return base+(gain[:,None]*weights)

    def full_vocab_bonus(self,q,keys,token_ids):
        if token_ids.numel()==0:return None
        attn,gain=self.attention(q[None],keys[None],
                                 torch.ones((1,len(token_ids)),
                                            dtype=torch.bool,device=q.device))
        size=49152  # fixed is NOT used: actual vocab dimension comes from caller.
        return attn[0]*gain[0]

WORD_CACHE={}
def candidate_positions(tok,prompt_ids):
    kept=[]
    for i,token in enumerate(prompt_ids):
        token=int(token)
        if token not in WORD_CACHE:
            value=tok.decode([token],skip_special_tokens=True).strip()
            WORD_CACHE[token]=len(value)>=2 and value.isalpha()
        if WORD_CACHE[token]:kept.append(i)
    return kept

def newly_authored_train(tok):
    accepted=[n for n in NOUNS if all(len(tok.encode(f"{c} {n}",add_special_tokens=False))==2
            for c in s30.TRAIN_ELIGIBLE)]
    if len(accepted)<20:
        raise AssertionError("S82 insufficient genuine tokenizer-compatible noun variety")
    # All words in the optional new TRAIN corpus are authored for this
    # new step, not borrowed from any embargoed independent QA.
    words=accepted[:24]
    result=[]
    for color in ("red","green","blue","yellow"):
        for j in range(8):
            x=f"{color} {words[j]}"
            y=f"{color} {words[(j+12)%len(words)]}"
            for i,frame in enumerate(TRAIN_FRAMES):
                a,b=TRAIN_NAMES[i]
                left=tokenize_one(tok,frame.format(a=a,b=b,left=x,right=y),x)
                right=tokenize_one(tok,frame.format(a=a,b=b,left=y,right=x),y)
                if left["first_expected_id"]!=right["first_expected_id"]:
                    raise AssertionError("S82 TRAIN swapped noun first-token inconsistency")
                result.append({"pair_id":82000+len(result),"kind":"TRAIN_new_copy",
                               "left":left,"right":right})
    if len(result)!=128 or len({p[s]["question"] for p in result for s in ("left","right")})!=256:
        raise AssertionError("S82 authored TRAIN corpus changed")
    return result,words

def full_gpu_generate(dense,first,later,head,tok,question,pointer=None):
    prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                   tokenize=False,add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    ids=torch.tensor([context],device="mps",dtype=torch.long)
    kept=candidate_positions(tok,context) if pointer is not None else []
    key_ids=torch.tensor([context[i] for i in kept],device="mps",dtype=torch.long)
    tokens=[]
    past=None
    stored=None
    t=time.perf_counter()
    with torch.inference_mode():
        for step in range(s56.MAX_NEW):
            inp=ids if step==0 else torch.tensor([[tokens[-1]]],device="mps",dtype=torch.long)
            out=dense(input_ids=inp,past_key_values=past,
                      use_cache=True,output_hidden_states=True)
            past=out.past_key_values
            if past is None or out.hidden_states[-1].device.type!="mps":
                raise AssertionError("S82 requires physical MPS KV-cache and hidden")
            hidden=out.hidden_states[-1][0]
            if step==0 and pointer is not None and kept:
                stored=hidden[torch.tensor(kept,device="mps")].detach()
            h=hidden[-1]
            raw_logits=out.logits[0,-1]
            main=first if step==0 else later
            logits=raw_logits+F.linear(main(h),head)
            if step==1 and pointer is not None and stored is not None and len(kept):
                # Preserve pre-existing EOS: generic pointer helps COPY, not
                # arbitrary continuation after a model has terminated.
                if int(logits.argmax())!=tok.eos_token_id:
                    values=pointer.full_vocab_bonus(h,stored,key_ids)
                    extra=torch.zeros_like(logits).index_add(0,key_ids,values)
                    logits=logits+extra
            token=int(logits.argmax())
            tokens.append(token)
            del out
            if token==tok.eos_token_id:break
    torch.mps.synchronize()
    return tokens,time.perf_counter()-t

def full_cpu_generate(model,sources,first,later,head,tok,question,pointer=None):
    prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                   tokenize=False,add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    ids=torch.tensor([context],dtype=torch.long)
    kept=candidate_positions(tok,context) if pointer is not None else []
    token_ids=torch.tensor([context[i] for i in kept],dtype=torch.long)
    tokens=[]
    with torch.inference_mode():
        for step in range(s56.MAX_NEW):
            clear(sources)
            out=model(input_ids=ids,use_cache=False,output_hidden_states=True)
            h=out.hidden_states[-1][0,-1]
            logits=out.logits[0,-1]+F.linear(
                first(h) if step==0 else later(h),head)
            if step==1 and pointer is not None and kept:
                if int(logits.argmax())!=tok.eos_token_id:
                    # Reconstructed source CPU produces all prompt states.
                    keys=out.hidden_states[-1][0,torch.tensor(kept)]
                    values=pointer.full_vocab_bonus(h,keys,token_ids)
                    logits=logits+torch.zeros_like(logits).index_add(0,token_ids,values)
            nextid=int(logits.argmax())
            tokens.append(nextid)
            del out
            clear(sources)
            if nextid==tok.eos_token_id:break
            ids=torch.cat((ids,torch.tensor([[nextid]])),dim=1)
    return tokens

def graded(toks,gold,tok):
    expected=tok.encode(gold,add_special_tokens=False)+[tok.eos_token_id]
    return {"actual":toks,"full_correct":toks==expected,
            "first_gold_correct":bool(toks) and toks[0]==expected[0]}

def all_gpu_probe(dense,first,later,head,tok,pairs,pointer=None,previous=None,abort_early=False):
    out={}
    for family,rows in pairs.items():
        family_out={}
        for pair in rows:
            for side in ("left","right"):
                key=f'{pair["pair_id"]}_{side}'
                sample=pair[side]
                generated,_=full_gpu_generate(dense,first,later,head,tok,sample["question"],pointer)
                rec=graded(generated,sample["answer"],tok)
                family_out[key]=rec
                if abort_early and previous is not None:
                    initial=previous[family][key]
                    if (initial["full_correct"] and not rec["full_correct"]) or (
                        initial["first_gold_correct"] and not rec["first_gold_correct"]):
                        return {"early_loss":(family,key),"rows":out}
        out[family]=family_out
    return {"early_loss":None,"rows":out}

def summarise(rows):
    return {family:{"full":sum(r["full_correct"] for r in cases.values()),
                    "first":sum(r["first_gold_correct"] for r in cases.values()),
                    "count":len(cases)}
            for family,cases in rows.items()}

@torch.no_grad()
def original_source_gpu_training_cache(dense,head,tok,later,pairs,baseline):
    """Extract all new next-token generic visual/context copies from REAL
    frozen 36M on MPS. Only complete two-word TRAIN token labels.
    """
    records=[]
    per={}
    for family,all_pairs in pairs.items():
        used=0;eligible=0
        for pair in all_pairs:
            for side in ("left","right"):
                item=pair[side]
                reply=tok.encode(item["answer"],add_special_tokens=False)
                if len(reply)!=2:continue
                prompt=item["prompt_ids"].tolist()
                positions=candidate_positions(tok,prompt)
                target=reply[1]
                if target not in [prompt[i] for i in positions]:continue
                eligible+=1
                model_ids=prompt+[reply[0]]
                seq=torch.tensor([model_ids],device="mps",dtype=torch.long)
                out=dense(input_ids=seq,output_hidden_states=True,use_cache=False)
                states=out.hidden_states[-1][0]
                k=states[torch.tensor(positions,device="mps")].detach()
                h=states[-1].detach()
                base=out.logits[0,-1]+F.linear(later(h),head)
                source_ids=[prompt[i] for i in positions]
                # Unique candidate vocabulary: all ordinary prompt words,
                # gold label and genuine strongest frozen-vocabulary rivals.
                selected=list(dict.fromkeys(source_ids))
                for idx in base.topk(16).indices.tolist():
                    if idx not in selected:selected.append(int(idx))
                if target not in selected:raise AssertionError("S82 target copying impossible")
                if len(selected)>110 or len(positions)>96:
                    raise AssertionError("S82 TRAIN source candidates out of bounds")
                cand=torch.tensor(selected,device="mps",dtype=torch.long)
                pos_ids=torch.tensor(source_ids,device="mps",dtype=torch.long)
                association=(pos_ids[:,None]==cand[None,:]).float()
                base_candidates=base.index_select(0,cand).detach()
                gold_col=selected.index(target)
                old=baseline[family][f'{pair["pair_id"]}_{side}']
                records.append({
                    "family":family,"query":h,"keys":k,
                    "association":association,"base":base_candidates,
                    "gold":gold_col,
                    "old_protected":bool(old["full_correct"]),
                    "old_marg":
                      float(base_candidates[gold_col]-torch.cat((
                        base_candidates[:gold_col],base_candidates[gold_col+1:]
                      )).max())})
                used+=1
                del out
        per[family]={"full_queries":len(all_pairs)*2,
                     "copyable_two_token_noun_examples":eligible,
                     "stored_training_records":used}
        print(json.dumps({"event":"S82_FULL_MODEL_GPU_GENERIC_POINTER_TRAIN_CACHE",
            "family":family,**per[family]}),flush=True)
    if not records or per["S82_new"]["stored_training_records"]<170:
        raise AssertionError("S82 insufficient novel diverse copyable TRAIN noun targets")
    return records,per

def batch_tensors(records):
    P=max(len(r["keys"]) for r in records)
    C=max(len(r["base"]) for r in records)
    B=len(records)
    device="mps"
    query=torch.stack([r["query"] for r in records])
    keys=torch.zeros((B,P,320),dtype=torch.float32,device=device)
    mask=torch.zeros((B,P),dtype=torch.bool,device=device)
    assoc=torch.zeros((B,P,C),dtype=torch.float32,device=device)
    base=torch.full((B,C),-1.e4,dtype=torch.float32,device=device)
    gold=torch.zeros(B,dtype=torch.long,device=device)
    protect=torch.zeros(B,dtype=torch.bool,device=device)
    old_margin=torch.zeros(B,dtype=torch.float32,device=device)
    for i,r in enumerate(records):
        n=len(r["keys"]);m=len(r["base"])
        keys[i,:n]=r["keys"]
        mask[i,:n]=True
        assoc[i,:n,:m]=r["association"]
        base[i,:m]=r["base"]
        gold[i]=r["gold"]
        protect[i]=r["old_protected"]
        old_margin[i]=r["old_marg"]
    return {"query":query,"keys":keys,"mask":mask,"assoc":assoc,
            "base":base,"gold":gold,
            "protected":protect,"prior_margin":old_margin}

def select(data,idx):
    return {k:v.index_select(0,idx) for k,v in data.items()}

def batch_scores(pointer,data):
    return pointer.candidate_scores(
        data["query"],data["keys"],data["mask"],
        data["assoc"],data["base"])

def train_loss(pointer,fresh,protected):
    logits=batch_scores(pointer,fresh)
    target=fresh["gold"]
    useful=F.cross_entropy(logits,target)
    keep=batch_scores(pointer,protected)
    gold=keep.gather(1,protected["gold"][:,None]).squeeze(1)
    rivals=keep.masked_fill(
        F.one_hot(protected["gold"],num_classes=keep.shape[1]).bool(),
        -1.e4).max(dim=1).values
    now_margin=gold-rivals
    regression=F.relu(protected["prior_margin"]+.02-now_margin)
    return useful+5.0*regression.square().mean(), {
        "train_only_new_copy_crossentropy":float(useful.detach()),
        "all_old_margin_regression_max":float(regression.detach().max()),
    }

def build_cpu_state(raw,weights):
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    f,l=s78.controllers(weights)
    return model,sources,head,f,l

def paid_manifest(original_envelope,original_side,new_side,report):
    m={
      "stage":"CALIBRI_S82_REAL_MPS_UNIVERSAL_COPY_POINTER_TRAIN_ONLY",
      "S74_original_frozen_controller_SHA256":sha(original_side),
      "S82_full_new_controller_SHA256":sha(new_side),
      "original_S61_physically_compressed_source_SHA256":sha(original_envelope),
      "original_source_bytes":len(original_envelope),
      "S82_complete_neural_controller_bytes":len(new_side),
      "S82_train_only_full_guard_scores":report["accepted_native_CPU_safety"],
      "all_previous_QA_including_S79_permanently_embargoed":True,
      "explicitly_NO_FP32_or_unseen_QA_claim":True,
      "no_LIVE_changes":True,
    }
    for _ in range(32):
        b=json.dumps(m,indent=2,sort_keys=True).encode()
        fields={"actual_manifest_bytes":len(b),
                "all_inclusive_physical_bytes":len(original_envelope)+len(new_side)+len(b),
                "all_inclusive_bpp":f"{(len(original_envelope)+len(new_side)+len(b))*8/SOURCE_PARAMS:.12f}"}
        if all(m.get(k)==v for k,v in fields.items()):break
        m.update(fields)
    b=json.dumps(m,indent=2,sort_keys=True).encode()
    if m["all_inclusive_physical_bytes"]!=len(original_envelope)+len(new_side)+len(b):
        raise AssertionError("S82 exact physical manifest size nonconvergence")
    if m["all_inclusive_physical_bytes"]*8/SOURCE_PARAMS>.7:
        raise AssertionError("S82 paid all neural weights exceed 0.7 bpp")
    return b,m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("S45_immutable_CB59",type=Path)
    ap.add_argument("S74_frozen_neural_controller",type=Path)
    ap.add_argument("S74_fully_paid_manifest",type=Path)
    ap.add_argument("S61_real_physical_compressed_source",type=Path)
    ap.add_argument("output_dir",type=Path)
    a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S82 requires actual Apple Silicon Metal GPU; CPU fallback forbidden")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    rng=random.Random(SEED)
    raw=a.S45_immutable_CB59.read_bytes()
    frozen=a.S74_frozen_neural_controller.read_bytes()
    compressed=a.S61_real_physical_compressed_source.read_bytes()
    previous=json.loads(a.S74_fully_paid_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        previous["S74_GPU_full_controller_SHA256"]!=sha(frozen) or
        previous["S71_frozen_controller_SHA256"]!=s73.S71_SHA or
        previous["selected_first_restoration_STEP_TRAIN_only"]!=48 or
        previous["S61_unchanged_exact_compressed_source_SHA256"]!=sha(compressed) or
        previous["physical_all_inclusive_bytes"]!=len(compressed)+len(frozen)+a.S74_fully_paid_manifest.stat().st_size or
        float(previous["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S82 model must be immutable fully physically paid S74")
    old_weights=load_file(str(a.S74_frozen_neural_controller))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    fresh,training_words=newly_authored_train(tok)
    pairs={
      "original":s30.make_training_pairs(tok),
      "augmented":s54.build_new_train(tok),
      "history":build_pairs(tok,"TRAIN")[:4],
      "S69":s69.build_new_train(tok),
      "S73":s73.new_s73_train(tok),
      "S82_new":fresh,
    }
    if [len(rows)*2 for rows in pairs.values()]!=[64,64,8,32,32,256]:
        raise AssertionError("S82 old/new TRAIN provenance changed")
    fingerprint=sha(json.dumps([
        {"cohort":name,"questions":[p[side]["question"] for side in ("left","right")]}
        for name,pairs_ in pairs.items() for p in pairs_],sort_keys=True).encode())
    cpu_model,source,head_cpu,cpu_first,cpu_later=build_cpu_state(raw,old_weights)
    dense=s30.dense_from_paid(CB59(raw))
    if sum(p.numel() for p in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S82 GPU full frozen model is not identical 36M architecture")
    dense=dense.to("mps").eval().requires_grad_(False)
    gpu_head=dense.lm_head.weight.detach()
    gpu_first,gpu_later=s78.controllers(old_weights)
    gpu_first=gpu_first.to("mps").eval().requires_grad_(False)
    gpu_later=gpu_later.to("mps").eval().requires_grad_(False)
    before=all_gpu_probe(dense,gpu_first,gpu_later,gpu_head,tok,pairs)["rows"]
    summary_before=summarise(before)
    names=["original","augmented","history","S69","S73"]
    if tuple(summary_before[n]["full"] for n in names)!=BASE_S74_FULL_TRAIN or (
       tuple(summary_before[n]["first"] for n in names)!=BASE_S74_FIRST_TRAIN):
        raise AssertionError("S82 actual frozen S74 old full or first TRAIN baseline drift")
    sample_pairs=pairs["original"][:4]
    for p in sample_pairs:
        for side in ("left","right"):
            q=p[side]["question"]
            cpu_ids=s56.greedy_first_active(cpu_model,source,cpu_first,cpu_later,head_cpu,tok,q)
            gpu_ids=before["original"][f'{p["pair_id"]}_{side}']["actual"]
            if cpu_ids!=gpu_ids:raise AssertionError("S82 frozen source GPU/CPU native token parity invalid")
    print(json.dumps({"event":"S82_FROZEN_NATIVE_S74_PROTECTED_BASELINE",
        "counts":summary_before,"new_TRAIN_vocab_count":len(training_words),
        "TRAIN_provenance_sha256":fingerprint,
        "GPU_CPU_S74_real_full_token_parity_8_of_8":True}),flush=True)
    cache,coverage=original_source_gpu_training_cache(dense,gpu_head,tok,gpu_later,pairs,before)
    combined=batch_tensors(cache)
    protected_ids=[i for i,r in enumerate(cache) if r["old_protected"]]
    train_ids=[i for i,r in enumerate(cache) if not r["old_protected"] and r["family"]!="history"]
    if len(protected_ids)<60 or len(train_ids)<120:
        raise AssertionError("S82 needs full old and genuinely wrong TRAIN contexts")
    protected=select(combined,torch.tensor(protected_ids,device="mps"))
    pointer=GenericContextPointer().to("mps")
    new_params=[p for p in pointer.parameters() if p.requires_grad]
    if not new_params or any(p.device.type!="mps" for p in new_params):
        raise AssertionError("S82 generic learned pointer gradients not on physical MPS")
    opt=torch.optim.AdamW(new_params,lr=LR,weight_decay=.00001)
    best=None
    bestscore=summary_before["S82_new"]["full"]
    training_start=time.perf_counter()
    snapshots=[]
    verified_grad=False
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        indices=torch.tensor([train_ids[rng.randrange(len(train_ids))] for _ in range(64)],
                             device="mps",dtype=torch.long)
        fresh_batch=select(combined,indices)
        loss,diagnostics=train_loss(pointer,fresh_batch,protected)
        if not torch.isfinite(loss):raise AssertionError("S82 MPS pointer training gradient nonfinite")
        loss.backward()
        if not verified_grad:
            grads=[p.grad for p in new_params if p.grad is not None]
            if not grads or any(g.device.type!="mps" for g in grads):
                raise AssertionError("S82 actual pointer backprop not on MPS GPU")
            verified_grad=True
        torch.nn.utils.clip_grad_norm_(new_params,3.0,error_if_nonfinite=True)
        opt.step()
        if step not in CHECKS:continue
        pointer.eval()
        guarded=all_gpu_probe(dense,gpu_first,gpu_later,gpu_head,tok,pairs,pointer,previous=before,abort_early=True)
        if guarded["early_loss"] is not None:
            row={"step":step,"all_old_answers_kept":False,
                 "first_forgetting_identity":guarded["early_loss"],
                 "candidate_eligible":False,
                 "loss":float(loss.detach()),**diagnostics}
            snapshots.append(row)
            print(json.dumps({"event":"S82_REAL_GPU_POINTER_PROTECTED_TRAIN_CHECK",**row}),flush=True)
            pointer.train()
            continue
        graded_now=summarise(guarded["rows"])
        gain=graded_now["S82_new"]["full"]-bestscore
        eligible=gain>0
        row={"step":step,"all_old_answers_kept":True,
             "full_native_GPU_TRAIN":graded_now,
             "previous_new_best":bestscore,
             "new_copy_full_train_gain":gain,
             "candidate_eligible":eligible,
             "loss":float(loss.detach()),**diagnostics}
        snapshots.append(row)
        print(json.dumps({"event":"S82_REAL_GPU_POINTER_PROTECTED_TRAIN_CHECK",**row}),flush=True)
        if eligible:
            bestscore=graded_now["S82_new"]["full"]
            best={"step":step,"state":{k:v.detach().cpu().clone()
                                    for k,v in pointer.state_dict().items()},
                  "gpu_native_TRAIN":graded_now}
        pointer.train()
    torch.mps.synchronize()
    elapsed=time.perf_counter()-training_start
    result={
       "stage":"CALIBRI_S82_REAL_MPS_GENERIC_TOKEN_COPY_POINTER_TRAIN_ONLY",
       "fully_paid_S74_source_SHA256":sha(frozen),
       "original_S74_all_inclusive_bpp":previous["physical_all_inclusive_bpp"],
       "real_frozen_36M_on_MPS":True,
       "real_pointer_GPU_gradients_proven":verified_grad,
       "new_neural_pointer_trainable_parameters":sum(p.numel() for p in new_params),
       "no_runtime_source_text_lookup_rules":True,
       "new_authored_train_nouns":training_words,
       "TRAIN_origin_sha256":fingerprint,
       "per_family_real_36M_GPU_copyable_supervised_context_coverage":coverage,
       "old_real_native_16_decoder_validation":True,
       "original_full_answer_TRAIN":summary_before,
       "TRAIN_research_only_full_Greedy_probes":snapshots,
       "gpu_train_plus_4_complete_old_native_guards_seconds":elapsed,
       "eligible_TRAIN_only_candidate":best is not None,
       "S79_S72_and_all_prior_QA_permanently_embargoed":True,
       "new_unseen_QA_conducted":False,
       "no_FP32_parity_claim":True,
       "no_Jessica_chat_or_LIVE_modified":True,
    }
    if best:
        pointer_cpu=GenericContextPointer()
        pointer_cpu.load_state_dict(best["state"],strict=True)
        pointer_cpu.eval().requires_grad_(False)
        # ALL old and new TRAIN full-sequence CPU validation,
        # independent of MPS floating-point choices.
        native={}
        losses=[]
        for family,rows in pairs.items():
            target={}
            for p in rows:
                for side in ("left","right"):
                    k=f'{p["pair_id"]}_{side}'
                    item=p[side]
                    generated=full_cpu_generate(cpu_model,source,cpu_first,cpu_later,
                                                head_cpu,tok,item["question"],pointer_cpu)
                    actual=graded(generated,item["answer"],tok)
                    old=before[family][k]
                    if (old["full_correct"] and not actual["full_correct"]) or (
                        old["first_gold_correct"] and not actual["first_gold_correct"]):
                        losses.append((family,k))
                    target[k]=actual
            native[family]=target
        native_summary=summarise(native)
        exact_match=all(native_summary[k]==best["gpu_native_TRAIN"][k] for k in pairs)
        accepted=(not losses and exact_match and
          native_summary["S82_new"]["full"]>summary_before["S82_new"]["full"])
        result["accepted_native_CPU_safety"]={
            "all_previously_right_FULL_and_FIRST_TRAIN_preserved":not losses,
            "lost_old_identity_keys":losses,
            "CPU_gpu_TRUSTED_TRUTH_COUNT_parity":exact_match,
            "native_CPU_TRAIN_counts":native_summary}
        if accepted:
            samples=make_samples(tok)
            cpu_times={}
            for label,ptr in (("S74",None),("S82",pointer_cpu)):
                durations=[]
                for _,_,question,_ in samples:
                    start=time.perf_counter()
                    full_cpu_generate(cpu_model,source,cpu_first,cpu_later,
                                      head_cpu,tok,question,ptr)
                    durations.append(time.perf_counter()-start)
                cpu_times[label]=statistics.median(durations)
            ratio=cpu_times["S74"]/cpu_times["S82"]
            result["CPU_speed_ratio_S82_to_S74"]=ratio
            if ratio>=.90:
                paid=dict(old_weights)
                for k,v in best["state"].items():
                    paid["generic_pointer."+k]=v.detach().contiguous()
                path=a.output_dir/"S82_REAL_GPU_GENERAL_TOKEN_POINTER_PAID_TRAIN_CANDIDATE.safetensors"
                save_file(paid,str(path))
                loaded=load_file(str(path))
                if any(not torch.equal(v,loaded[k]) for k,v in old_weights.items()):
                    raise AssertionError("S82 S74 original paid weights changed")
                manifest,m=paid_manifest(compressed,frozen,path.read_bytes(),result)
                (a.output_dir/"S82_REAL_GPU_ALL_INCLUSIVE_PAID_MANIFEST.json").write_bytes(manifest)
                result["accepted_TRAIN_only_candidate"]=True
                result["physical_bpp"]=m["all_inclusive_bpp"]
                result["selected_real_GPU_TRAIN_step"]=best["step"]
                result["saved_controller_SHA256"]=sha(path.read_bytes())
            else:result["accepted_TRAIN_only_candidate"]=False
        else:result["accepted_TRAIN_only_candidate"]=False
    else:result["accepted_TRAIN_only_candidate"]=False
    (a.output_dir/"S82_REAL_GPU_GENERIC_NEURAL_POINTER_TRAIN_ONLY_REPORT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S82_FINAL_GENERIC_POINTER_REAL_GPU_TRAIN_ONLY",
        "real_gpu_gradients":verified_grad,
        "candidate_accepted":result["accepted_TRAIN_only_candidate"],
        "all_inclusive_bpp":result.get("physical_bpp"),
        "original_TRAIN":summary_before,
        "max_new_full_correct_TRAIN":bestscore,
        "S79_and_all_other_inspected_QA_never_read":True,
        "no_unseen_generalization_claim":True}),flush=True)

if __name__=="__main__":main()
