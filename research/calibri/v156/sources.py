"""Original source TRAIN split: third untouched WikiText and AG News regions.

Read nothing from official v89 TEST or SciQ/BoolQ labels. This new source
is never used to fit or retune the physically paid 1-bit source operators.
"""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (WIKI_REVISION,NEWS_REVISION,WIKI_START,NEWS_START,
                       NEWS_END,TOKENS,DOMAINS,check)

def load_third_holdout():
    check()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    raw=load_token_ids(tok,"train",WIKI_REVISION,WIKI_START+TOKENS)
    wiki=raw[WIKI_START:WIKI_START+TOKENS].clone()
    news_rows=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                           split=f"train[{NEWS_START}:{NEWS_END}]")
    text="\n\n".join(str(r["text"]) for r in news_rows)
    news=tok(text,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKENS].clone()
    if len(wiki)!=TOKENS or len(news)!=TOKENS:
        raise ValueError("frozen third source holdout has insufficient original tokens")
    data=dict(zip(DOMAINS,(wiki,news)))
    provenance={
        DOMAINS[0]:{"dataset":"Salesforce/wikitext","split":"train",
                    "revision":WIKI_REVISION,"token_start":WIKI_START,
                    "token_end":WIKI_START+TOKENS,
                    "sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        DOMAINS[1]:{"dataset":"fancyzhx/ag_news","split":"train",
                    "revision":NEWS_REVISION,"record_start":NEWS_START,
                    "record_end":NEWS_END,"no_labels":True,
                    "sha256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }
    return data,provenance
