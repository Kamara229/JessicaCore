"""CALIBRI2 S8 — capacity-use training of S7 36M physically paid 5-bit model.

Purpose: test hypothesis that broad TRAIN-only compositional pairs plus actual
WikiText TRAIN language modelling and lower LR prevent the repeated S4/S6
overfit to 24+64+56 fixed question templates. Nothing here reads QA/VALID.
Predeclare one training schedule. The outputs are a research experiment, NOT
a claim of FP32 language understanding or automatic bigger-model promotion.

No FP32 pretrained or teacher weights, no LIVE Jessica or Render/Supabase/APK.
All 85 physically paid K32/V8 centroids and occasional 5bit indices train.
New compact weights must serialize into ONE identical-sized .cb59 archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes, apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs, tokenize_one
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from quality import DATASET, CONFIG
from v174.run import save

SOURCE_SHA = "70512bc374c1846631f635998f9e035e2e25f147da185c2b7b0dd663799d9f8a"
SOURCE_BYTES = 3141513
SOURCE_PARAMS = 36380480
PAID_OPERATORS = 85
SEED = 2080
STEPS = 768
LR = .0012
CONTEXT = 48
MAX_CODES = 512
MODEL_SPEC = ("36m_320x12_ffn1536", 320, 1536, 12, 5, 1)

# These exact two-word combinations are from TRAIN only. The post-seal fresh
# benchmark has separate words/phrases and never participates in this file.
LEFT_ADJ = ("amber", "wooden", "gentle", "tall", "smooth", "bright",
            "distant", "silent", "cool", "copper", "hidden", "narrow",
            "fresh", "plain", "early", "golden")
LEFT_NOUN = ("lantern", "river", "pillow", "bridge", "valley", "forest",
             "basket", "castle", "meadow", "window", "garden", "island",
             "engine", "sailor", "harbor", "mountain")
RIGHT_ADJ = ("quiet", "frosted", "little", "purple", "winter", "yellow",
             "silver", "open", "brave", "light", "small", "warm",
             "shining", "soft", "strong", "clear")
RIGHT_NOUN = ("tunnel", "ribbon", "planet", "notebook", "station", "signal",
              "violin", "bicycle", "triangle", "flower", "rabbit", "shell",
              "coffee", "beacon", "ocean", "harvest")

SENTENCE_CONTRASTS = (
 ("What is a lantern useful for?", "Lanterns produce light.",
  "What is a bridge useful for?", "Bridges connect places."),
 ("Why do people use blankets?", "Blankets help people stay warm.",
  "Why do people use maps?", "Maps help people find places."),
 ("What can books provide?", "Books provide helpful information.",
  "What can clocks provide?", "Clocks provide the time."),
 ("How do fish breathe?", "Fish absorb oxygen in water.",
  "How do birds fly?", "Birds use their wings."),
 ("What can a pencil do?", "Pencils create written marks.",
  "What can a bicycle do?", "Bicycles carry a rider."),
 ("Why do people wash hands?", "Washing removes unwanted germs.",
  "Why do people wear shoes?", "Shoes protect people's feet."),
 ("What is snow?", "Snow is frozen water.",
  "What is a rainbow?", "Rainbows contain colorful light."),
 ("What does a plant require?", "Plants need water and light.",
  "What does a vehicle require?", "Vehicles need energy to move."),
)

def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def make_broad_train(tok):
    data = []
    for i in range(320):
        j = i // 5
        group = i % 5
        l = LEFT_ADJ[j % 16] + " " + LEFT_NOUN[(j*3 + j//16) % 16]
        r = RIGHT_ADJ[j % 16] + " " + RIGHT_NOUN[(j*5 + j//16) % 16]
        name = f"TRAINREPLAY{i:03d}"
        if group == 0:
            left = (f"A training note {name} says '{l}'. Copy its two-word phrase exactly.", l)
            right = (f"A training note {name} says '{r}'. Copy its two-word phrase exactly.", r)
            domain = "train_copy_two"
        elif group == 1:
            leftphrase = l + " light"
            rightphrase = r + " stone"
            left = (f"Training note {name} contains '{leftphrase}'. Repeat all three words.", leftphrase)
            right = (f"Training note {name} contains '{rightphrase}'. Repeat all three words.", rightphrase)
            domain = "train_copy_three"
        elif group == 2:
            owner = f"TRAINHaru{i:03d}"
            other = f"TRAINVero{i:03d}"
            left = (f"In a made-up scene, {owner} keeps a red folder, "
                    f"and {other} keeps a blue folder. "
                    f"Which color belongs to {owner}? Answer one word.", "red")
            right = (f"In a made-up scene, {owner} keeps a blue folder, "
                     f"and {other} keeps a red folder. "
                     f"Which color belongs to {owner}? Answer one word.", "blue")
            domain = "train_binding"
        elif group == 3:
            base = 10103 + i*31
            diff = 3 + i % 17
            left = (f"In a TRAIN number game, is {base+diff} greater than {base}? "
                    "Answer Yes or No.", "Yes")
            right = (f"In a TRAIN number game, is {base} greater than {base+diff}? "
                     "Answer Yes or No.", "No")
            domain = "train_numeric"
        else:
            q1,a1,q2,a2 = SENTENCE_CONTRASTS[j % len(SENTENCE_CONTRASTS)]
            left = (f"Training practice {name}. {q1} Reply with one sentence.", a1)
            right = (f"Training practice {name}. {q2} Reply with one sentence.", a2)
            domain = "train_sentence"
        lobj = tokenize_one(tok, *left)
        robj = tokenize_one(tok, *right)
        if lobj["first_expected_id"] == robj["first_expected_id"]:
            raise AssertionError(f"Unbalanced TRAIN first token on pair {i}")
        data.append({"split":"TRAIN", "pair_id":144+i,
                     "domain":domain, "left":lobj, "right":robj})
    unique_questions = [p[s]["question"] for p in data for s in ("left","right")]
    if len(data)!=320 or len(set(unique_questions))!=640:
        raise AssertionError("TRAIN broad source is not diverse and unique")
    return data

def freeze_language_corpus(tok):
    # Independently permitted, real public TRAIN text, never validation or test.
    revision = HfApi().dataset_info(DATASET).sha
    lines = load_dataset(DATASET,CONFIG,split="train[:1000]",revision=revision,
                         trust_remote_code=False)
    text = "\n\n".join(row["text"] for row in lines if row["text"].strip())
    tokens = tok.encode(text, add_special_tokens=False)
    if len(tokens) < CONTEXT*192:
        raise AssertionError("Not enough actual TRAIN tokens")
    ids = torch.tensor(tokens[:CONTEXT*192],dtype=torch.long)
    source_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids,{"dataset":DATASET, "config":CONFIG,
                "train_revision":revision,"split":"train[:1000]",
                "token_limit":len(ids),"train_tokens_sha":source_sha,
                "never_QA_or_test":True}

def weighted_pair_loss(model, pair):
    scores=[]
    first=[]
    target=[]
    multi = "copy" in pair["domain"] or "sentence" in pair["domain"]
    for side in ("left","right"):
        q = pair[side]
        logits = model(input_ids=q["ids"][None],use_cache=False).logits[0][q["positions"]]
        tgt=q["targets"]
        ce=F.cross_entropy(logits,tgt,reduction="none")
        # Preserve all answer tokens including EOS, not only the first word.
        weights=torch.ones_like(ce)
        if multi and len(weights)>2:
            weights[1:-1]=2.0
        if multi:
            weights[-1]=1.4
        scores.append((weights*ce).sum()/weights.sum())
        first.append(logits[0])
        target.append(int(tgt[0]))
    if target[0]==target[1]:
        raise AssertionError("TRAIN contrast lacks opposing first token")
    gaps=[first[k][target[k]]-first[k][target[1-k]] for k in (0,1)]
    margin=sum(F.relu(1.0-z) for z in gaps)/2
    return sum(scores)/2 + .15*margin

def train_score(model,sources,pairs):
    r=probe_greedy_train(model,sources,pairs,max_new_tokens=12)
    rows=[]
    for p in pairs:
        for side in ("left","right"):
            q=r[p["pair_id"]]["sides"][side]
            rows.append({"pair":p["pair_id"],"domain":p["domain"],"side":side,
                         "gold":p[side]["answer"],
                         "actual_tokens":q["generated_tokens"],
                         "correct_full_EOS":bool(q["exact_with_EOS"])})
    return {"correct":sum(v["correct_full_EOS"] for v in rows),
            "total":len(rows),"rows":rows}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("source",type=Path)
    a.add_argument("output",type=Path)
    args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    setup()
    raw=args.source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("Require exact verified 36M S7 source, not S4/S6 overfit")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Unexpected original unique parameter count")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    old=build_pairs(tok,"TRAIN") + s3_pairs(tok) + s4_pairs(tok)
    new=make_broad_train(tok)
    if len(old)!=144:
        raise AssertionError("Historic 144 TRAIN pairs missing")
    full={p[side]["question"] for p in old+new for side in ("left","right")}
    if len(full)!=2*(len(old)+len(new)):
        raise AssertionError("TRAIN source duplicate or wrong split")
    corpus,provenance=freeze_language_corpus(tok)
    cfg=build_config(MODEL_SPEC)
    model,sources=create_small_paid_model(archive,cfg)
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("All 85 physically paid 8D source matrices required")
    precommitted_train_probe=old[:8]+new[:8]
    baseline=train_score(model,sources,precommitted_train_probe)
    optimizer=torch.optim.Adam([s.centroids for s in sources.values()],lr=LR)
    progress=[]
    counts=Counter()
    for step in range(STEPS):
        turn=step % 8
        if turn in (0,1):
            kind="historical_TRAIN_replay"
            pair=old[counts[kind]%len(old)]
            ids=None
        elif turn in (2,3,4,5):
            kind="broad_compositional_TRAIN"
            pair=new[counts[kind]%len(new)]
            ids=None
        else:
            kind="licensed_public_WikiText_TRAIN"
            pair=None
            offset=(counts[kind]%(len(corpus)//CONTEXT))*CONTEXT
            ids=corpus[offset:offset+CONTEXT]
        counts[kind]+=1
        tgt=sources[keys[step%len(keys)]]
        observed=[]
        def hook(_module,_inputs,weight):
            weight.retain_grad()
            observed.append(weight)
        handler=tgt.register_forward_hook(hook)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            if pair is not None:
                loss=weighted_pair_loss(model,pair)
            else:
                logits=model(input_ids=ids[None],use_cache=False).logits[0]
                loss=F.cross_entropy(logits[:-1],ids[1:])
            before=float(loss.detach())
            loss.backward()
        finally:
            handler.remove()
        if not observed or observed[-1].grad is None:
            raise AssertionError("No gradient to targeted physically paid 8D operator")
        grad=observed[-1].grad.detach().cpu().numpy().copy()
        if sum(s.centroids.grad is not None for s in sources.values())!=PAID_OPERATORS:
            raise AssertionError("Not all 85 compressed matrix codebooks had gradients")
        gn=float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in sources.values()],1.0,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():
                s.centroids.clamp_(-16,16)
        clear(sources)
        accepted=0
        if (step+1)%24==0:
            with torch.inference_mode():
                if pair is not None:
                    baseline_loss=float(weighted_pair_loss(model,pair))
                else:
                    base_logits=model(input_ids=ids[None],use_cache=False).logits[0]
                    baseline_loss=float(F.cross_entropy(base_logits[:-1],ids[1:]))
                source_matrix=tgt().detach().cpu().numpy().copy()
            proposal=propose_codes(tgt,grad,source_matrix)
            if proposal["proposed_groups"]>MAX_CODES:
                raise AssertionError("Changed paid 5-bit index proposal budget")
            if proposal["proposed_groups"]:
                with torch.no_grad():
                    apply_codes(tgt,proposal,True)
                clear(sources)
                with torch.inference_mode():
                    if pair is not None:
                        candidate=float(weighted_pair_loss(model,pair))
                    else:
                        cand_logits=model(input_ids=ids[None],use_cache=False).logits[0]
                        candidate=float(F.cross_entropy(cand_logits[:-1],ids[1:]))
                if np.isfinite(candidate) and candidate+1e-7<baseline_loss:
                    accepted=proposal["proposed_groups"]
                else:
                    with torch.no_grad():
                        apply_codes(tgt,proposal,False)
        clear(sources)
        progress.append({"step":step+1,"kind":kind,
                         "id":None if pair is None else pair["pair_id"],
                         "train_loss":before,"clipped_grad_norm":gn,
                         "accepted_5bit_indices":accepted})
        if (step+1)%128==0:
            print(json.dumps({"event":"CALIBRI2_S8_TRAIN_PROGRESS",
                "step":step+1,"loss":before,"kind":kind,
                "accepted_5bit_so_far":sum(z["accepted_5bit_indices"] for z in progress),
                "counts":dict(counts)}),flush=True)
    packed,info=replace_discrete(archive,sources,strict=False)
    output=args.output/"CALIBRI2_S8_36M_BROAD_TRAIN_PAID.cb59"
    output.write_bytes(packed)
    sha=hashlib.sha256(packed).hexdigest()
    if len(packed)!=SOURCE_BYTES or info["complete_bpp"]>.700 or sha==SOURCE_SHA:
        raise AssertionError("S8 physical archive growth or unchanged model")
    reloaded,reloaded_src=create_small_paid_model(CB59(packed),cfg)
    before_reload=train_score(model,sources,precommitted_train_probe)
    after_reload=train_score(reloaded,reloaded_src,precommitted_train_probe)
    if before_reload["rows"]!=after_reload["rows"]:
        raise AssertionError("Compressed archive changes full sequences upon reload")
    lost=sum(x["correct_full_EOS"] and not y["correct_full_EOS"]
             for x,y in zip(baseline["rows"],after_reload["rows"]))
    gained=sum(not x["correct_full_EOS"] and y["correct_full_EOS"]
               for x,y in zip(baseline["rows"],after_reload["rows"]))
    report={
        "stage":"CALIBRI2_S8_36M_DIVERSE_PLUS_REAL_TEXT",
        "source_S7_sha256":SOURCE_SHA,"result_sha256":sha,
        "parameters":SOURCE_PARAMS,"archive_bytes":len(packed),
        "physical_bpp":info["complete_bpp"],
        "all_85_paid_centroids_trained":True,
        "additional_5bit_swaps":sum(v["accepted_5bit_indices"] for v in progress),
        "fixed_steps":STEPS,"learning_rate":LR,
        "TRAIN_old_pairs":len(old),"TRAIN_broad_new_pairs":len(new),
        "recorded_actual_WikiText_TRAIN":provenance,
        "precommitted_curriculum_counts":dict(counts),
        "no_original_FP32_teacher_weights":True,
        "no_QA_VALID_TEST_or_QA_checkpoint_selection":True,
        "lower_LR_and_diverse_data_is_hypothesis_not_quality_claim":True,
        "train_regression_old_answers":baseline,
        "train_post_update_answers":after_reload,
        "TRAIN_old_full_responses_lost":lost,
        "TRAIN_old_full_responses_gained":gained,
        "physical_archive_reload_exact_full_answers":True,
        "optimization_trace":progress,
    }
    save(args.output/"ACTUAL_CALIBRI2_S8_DIVERSE_TRAIN.json",report)
    print(json.dumps({"event":"CALIBRI2_S8_SEALED",
        "sha256":sha,"bpp":info["complete_bpp"],
        "bytes":len(packed),"train_before":baseline["correct"],
        "train_after":after_reload["correct"],"old_lost":lost,"new_gained":gained,
        "total_train_probe":after_reload["total"]}),flush=True)

if __name__=="__main__":
    main()
