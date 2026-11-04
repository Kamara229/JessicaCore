"""Frozen new WikiText/news original TRAIN-split holdout, NOT official TEST."""
import hashlib
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from v147.protocol import WIKI_REVISION,NEWS_REVISION
from .protocol import (VALID_DOMAINS,VALID_TOKENS,WIKI_VALID_START,
                       NEWS_VALID_START,NEWS_VALID_END)


def load_new_valid():
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    wiki_all=load_token_ids(tok,"train",WIKI_REVISION,
                             WIKI_VALID_START+VALID_TOKENS)
    wiki=wiki_all[WIKI_VALID_START:WIKI_VALID_START+VALID_TOKENS].clone()
    news_rows=load_dataset("fancyzhx/ag_news",
                           revision=NEWS_REVISION,
                           split=f"train[{NEWS_VALID_START}:{NEWS_VALID_END}]")
    news_text="\n\n".join(str(r["text"]) for r in news_rows)
    news=tok(news_text,add_special_tokens=False,
             return_tensors="pt").input_ids[0,:VALID_TOKENS].clone()
    if len(wiki)!=VALID_TOKENS or len(news)!=VALID_TOKENS:
        raise ValueError("new untouched TRAIN VALID range incomplete")
    result={VALID_DOMAINS[0]:wiki,VALID_DOMAINS[1]:news}
    provenance={
        VALID_DOMAINS[0]:{
            "dataset":"Salesforce/wikitext","split":"train",
            "revision":WIKI_REVISION,
            "token_range":[WIKI_VALID_START,WIKI_VALID_START+VALID_TOKENS],
            "token_sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
        },
        VALID_DOMAINS[1]:{
            "dataset":"fancyzhx/ag_news","split":"train",
            "revision":NEWS_REVISION,
            "records":[NEWS_VALID_START,NEWS_VALID_END],
            "token_sha256":hashlib.sha256(news.numpy().tobytes()).hexdigest(),
            "text_only_no_supervised_labels":True,
        },
    }
    return result,provenance
