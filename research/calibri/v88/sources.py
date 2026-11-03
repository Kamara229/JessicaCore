"""Predeclared genuine ORIGINAL TRAIN-only WikiText + AGNews FIT/VALID.

No official TEST is touched; these token/row ranges are newly chosen.
Corpus labels only assign TRAIN and VALID, never enter runtime operator.
"""
from __future__ import annotations
import hashlib
from datasets import load_dataset
from huggingface_hub import HfApi
from quality import DATASET,load_token_ids

TOKENS=128
WINDOW=64
REGIONS={
 "wiki_fit":("wiki",25000),
 "news_fit":("news",10000),
 "wiki_valid":("wiki",27000),
 "news_valid":("news",10300),
}
def actual_original_sources(tok):
    revisions={
        "wiki":HfApi().dataset_info(DATASET).sha,
        "news":HfApi().dataset_info("fancyzhx/ag_news").sha,
    }
    parts={}
    for name,(kind,start) in REGIONS.items():
        if kind=="wiki":
            ids=load_token_ids(tok,"train",revisions[kind],start+TOKENS)[start:start+TOKENS]
            origin={"dataset":DATASET,"split":"train","revision":revisions[kind],
                    "original_token_range":[start,start+TOKENS]}
        else:
            end=start+40
            rows=load_dataset("fancyzhx/ag_news",split=f"train[{start}:{end}]",
                              revision=revisions[kind])
            raw="\n\n".join(str(row["text"]) for row in rows)
            ids=tok(raw,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKENS]
            origin={"dataset":"fancyzhx/ag_news","split":"train",
                    "revision":revisions[kind],"original_row_range":[start,end],
                    "text_only_no_labels":True}
        if ids.numel()!=TOKENS:
            raise AssertionError("original pinned TRAIN text insufficient")
        origin["token_sha256"]=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        parts[name]={"origin":origin,
                     "windows":[ids[0:WINDOW].clone(),ids[WINDOW:TOKENS].clone()]}
    if len({x["origin"]["token_sha256"] for x in parts.values()})!=len(parts):
        raise AssertionError("duplicate source examples")
    return parts
