"""New original TRAIN text windows, independent from earlier TRAIN diagnostic."""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (WIKI_REVISION,NEWS_REVISION,WIKI_START,
                       NEWS_START,NEWS_END,VALID_TOKENS,DOMAINS,check)


def fresh_valid():
    check()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    old=load_token_ids(tok,"train",WIKI_REVISION,WIKI_START+VALID_TOKENS)
    wiki=old[WIKI_START:WIKI_START+VALID_TOKENS].clone()
    news=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                      split=f"train[{NEWS_START}:{NEWS_END}]")
    plain="\n\n".join(str(r["text"]) for r in news)
    news_tokens=tok(plain,add_special_tokens=False,
                    return_tensors="pt").input_ids[0,:VALID_TOKENS].clone()
    if len(wiki)!=VALID_TOKENS or len(news_tokens)!=VALID_TOKENS:
        raise ValueError("fresh v160 validation original TRAIN source missing")
    rows=dict(zip(DOMAINS,(wiki,news_tokens)))
    evidence={
        DOMAINS[0]:{"dataset":"Salesforce/wikitext","split":"train",
                    "revision":WIKI_REVISION,
                    "token_offsets":[WIKI_START,WIKI_START+VALID_TOKENS],
                    "SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        DOMAINS[1]:{"dataset":"fancyzhx/ag_news","split":"train",
                    "revision":NEWS_REVISION,
                    "record_offsets":[NEWS_START,NEWS_END],
                    "no_labels":True,
                    "SHA256":hashlib.sha256(news_tokens.numpy().tobytes()).hexdigest()},
    }
    return rows,evidence
