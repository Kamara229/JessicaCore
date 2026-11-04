"""Fresh original TRAIN-only source; exact source revision held constant."""
import hashlib
from transformers import AutoTokenizer
from datasets import load_dataset
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (WIKI_REVISION,NEWS_REVISION,WIKI_START,NEWS_START,
                       NEWS_END,TOKENS,DOMAINS,check)

def new_source():
    check()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    wiki_all=load_token_ids(tokenizer,"train",WIKI_REVISION,WIKI_START+TOKENS)
    wiki=wiki_all[WIKI_START:WIKI_START+TOKENS].clone()
    records=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                         split=f"train[{NEWS_START}:{NEWS_END}]")
    news_text="\n\n".join(str(row["text"]) for row in records)
    news=tokenizer(news_text,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKENS].clone()
    if len(wiki)!=TOKENS or len(news)!=TOKENS:raise ValueError("insufficient new source tokens")
    sources={DOMAINS[0]:wiki,DOMAINS[1]:news}
    provenance={
      DOMAINS[0]:{"dataset":"Salesforce/wikitext","split":"train",
        "revision":WIKI_REVISION,"token_span":[WIKI_START,WIKI_START+TOKENS],
        "sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
      DOMAINS[1]:{"dataset":"fancyzhx/ag_news","split":"train",
        "revision":NEWS_REVISION,"record_span":[NEWS_START,NEWS_END],
        "sha256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }
    return sources,provenance
