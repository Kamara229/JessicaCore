"""New original source TRAIN text windows. NEVER original frozen v89 TEST."""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from quality import load_token_ids
from .protocol import (check,WIKI_REVISION,NEWS_REVISION,WIKI_START,
                       NEWS_START,NEWS_END,TOKENS,DOMAINS)


def unopened_until_paid_archive_frozen():
    check()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    all_text=load_token_ids(tok,"train",WIKI_REVISION,WIKI_START+TOKENS)
    wiki=all_text[WIKI_START:WIKI_START+TOKENS].clone()
    news_records=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,
                        split=f"train[{NEWS_START}:{NEWS_END}]")
    joined="\n\n".join(str(r["text"]) for r in news_records)
    news=tok(joined,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKENS].clone()
    if len(wiki)!=TOKENS or len(news)!=TOKENS:
        raise ValueError("fixed source slice does not contain enough tokens")
    values=dict(zip(DOMAINS,(wiki,news)))
    source={
       DOMAINS[0]:{"dataset":"Salesforce/wikitext","split":"train",
         "revision":WIKI_REVISION,
         "token_offsets":[WIKI_START,WIKI_START+TOKENS],
         "token_SHA256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest()},
       DOMAINS[1]:{"dataset":"fancyzhx/ag_news","split":"train",
         "revision":NEWS_REVISION,"record_offsets":[NEWS_START,NEWS_END],
         "token_SHA256":hashlib.sha256(news.numpy().tobytes()).hexdigest()},
    }
    return values,source
