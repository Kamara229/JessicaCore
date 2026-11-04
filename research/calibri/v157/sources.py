"""New long fourth original TRAIN-only text holdout, no v89 TEST, no QA."""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (WIKI_REVISION,NEWS_REVISION,WIKI_START,NEWS_START,
                       NEWS_END,TOKENS,DOMAINS,check)


def fourth_untouched_sources():
    check()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    source=load_token_ids(tok,"train",WIKI_REVISION,WIKI_START+TOKENS)
    wiki=source[WIKI_START:WIKI_START+TOKENS].clone()
    records=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                         split=f"train[{NEWS_START}:{NEWS_END}]")
    raw="\n\n".join(str(r["text"]) for r in records)
    news=tok(raw,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKENS].clone()
    if len(wiki)!=TOKENS or len(news)!=TOKENS:
        raise ValueError("fourth pristine original TRAIN source shorter than preregistered")
    out=dict(zip(DOMAINS,(wiki,news)))
    proof={
        DOMAINS[0]:{"source":"Salesforce/wikitext","revision":WIKI_REVISION,
                    "split":"train","token_interval":[WIKI_START,WIKI_START+TOKENS],
                    "token_SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        DOMAINS[1]:{"source":"fancyzhx/ag_news","revision":NEWS_REVISION,
                    "split":"train","record_interval":[NEWS_START,NEWS_END],
                    "first_token_count":TOKENS,
                    "token_SHA256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }
    return out,proof
