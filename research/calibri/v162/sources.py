"""Fifth, entirely disjoint original TRAIN-split natural-text source regions."""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from quality import load_token_ids
from binary_geometry_v64 import MODEL,MODEL_SHA
from .protocol import (check,DOMAINS,WIKI_REVISION,NEWS_REVISION,WIKI_START,
                       NEWS_START,NEWS_END,SOURCE_TOKENS)


def load_untouched_source():
    check()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    original=load_token_ids(tok,"train",WIKI_REVISION,WIKI_START+SOURCE_TOKENS)
    wiki=original[WIKI_START:WIKI_START+SOURCE_TOKENS].clone()
    data=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                      split=f"train[{NEWS_START}:{NEWS_END}]")
    text="\n\n".join(str(x["text"]) for x in data)
    news=tok(text,add_special_tokens=False,
             return_tensors="pt").input_ids[0,:SOURCE_TOKENS].clone()
    if len(wiki)!=SOURCE_TOKENS or len(news)!=SOURCE_TOKENS:
        raise ValueError("not enough original source tokens in fixed disjoint slice")
    sources=dict(zip(DOMAINS,(wiki,news)))
    origins={
        DOMAINS[0]:{"dataset":"Salesforce/wikitext","split":"train",
            "revision":WIKI_REVISION,
            "token_indices":[WIKI_START,WIKI_START+SOURCE_TOKENS],
            "exact_token_SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
        DOMAINS[1]:{"dataset":"fancyzhx/ag_news","split":"train",
            "revision":NEWS_REVISION,"rows":[NEWS_START,NEWS_END],
            "exact_token_SHA256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }
    return sources,origins
