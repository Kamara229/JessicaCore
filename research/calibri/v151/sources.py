"""Untouched preregistered original TRAIN-split segments, not v89 TEST."""
import hashlib
from transformers import AutoTokenizer
from datasets import load_dataset
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (WIKI_REVISION,NEWS_REVISION,WIKI_NEW_START,
                       NEWS_NEW_START,NEWS_NEW_END,NEW_VALID_DOMAINS,
                       NEW_VALID_TOKENS,check_protocol)


def new_unseen_valid():
    check_protocol()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    records=load_token_ids(tok,"train",WIKI_REVISION,
                           WIKI_NEW_START+NEW_VALID_TOKENS)
    wiki=records[WIKI_NEW_START:WIKI_NEW_START+NEW_VALID_TOKENS].clone()
    news=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                      split=f"train[{NEWS_NEW_START}:{NEWS_NEW_END}]")
    text="\n\n".join(str(x["text"]) for x in news)
    raw=tok(text,add_special_tokens=False,
            return_tensors="pt").input_ids[0,:NEW_VALID_TOKENS].clone()
    if len(wiki)!=NEW_VALID_TOKENS or len(raw)!=NEW_VALID_TOKENS:
        raise ValueError("valid source insufficient, not allowed to fill from TEST")
    src=dict(zip(NEW_VALID_DOMAINS,(wiki,raw)))
    references={
        NEW_VALID_DOMAINS[0]:{"dataset":"Salesforce/wikitext",
            "split":"train","revision":WIKI_REVISION,
            "token_range":[WIKI_NEW_START,WIKI_NEW_START+NEW_VALID_TOKENS],
            "sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        NEW_VALID_DOMAINS[1]:{"dataset":"fancyzhx/ag_news",
            "split":"train","revision":NEWS_REVISION,
            "record_range":[NEWS_NEW_START,NEWS_NEW_END],
            "sha256":hashlib.sha256(raw.numpy().tobytes()).hexdigest()}
    }
    return src,references
