"""Pinned original TRAIN split, new disjoint WikiText and news validation ranges.

These are neither the official v89 independent TEST, nor v144 selection/
confirmation samples; no examples or model labels participate in fitting.
"""
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from quality import DATASET,CONFIG,load_token_ids
from .protocol import (SOURCE_DOMAINS,TOKEN_COUNT,WIKI_REVISION,NEWS_REVISION,
                       WIKI_SOURCE_START,NEWS_SOURCE_START,NEWS_SOURCE_END,
                       verify_predeclared)


def load_new_train_holdout():
    verify_predeclared()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    wiki_all=load_token_ids(tokenizer,"train",WIKI_REVISION,
                            WIKI_SOURCE_START+TOKEN_COUNT)
    wiki=wiki_all[WIKI_SOURCE_START:WIKI_SOURCE_START+TOKEN_COUNT]
    news_rows=load_dataset("fancyzhx/ag_news",
                           split=f"train[{NEWS_SOURCE_START}:{NEWS_SOURCE_END}]",
                           revision=NEWS_REVISION)
    text="\n\n".join(str(x["text"]) for x in news_rows)
    news=tokenizer(text,add_special_tokens=False,return_tensors="pt").input_ids[0,:TOKEN_COUNT]
    if len(wiki)!=TOKEN_COUNT or len(news)!=TOKEN_COUNT:
        raise ValueError("predeclared text source too short")
    outputs={"wiki_shift":wiki,"news_shift":news}
    provenances={
      "wiki_shift":{
        "dataset":DATASET,
        "config":CONFIG,
        "split":"train",
        "revision":WIKI_REVISION,
        "tokens":[WIKI_SOURCE_START,WIKI_SOURCE_START+TOKEN_COUNT],
        "sha256":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
      },
      "news_shift":{
        "dataset":"fancyzhx/ag_news",
        "split":"train",
        "revision":NEWS_REVISION,
        "records":[NEWS_SOURCE_START,NEWS_SOURCE_END],
        "token_indices_within_records":[0,TOKEN_COUNT],
        "text_only_no_labels":True,
        "sha256":hashlib.sha256(news.numpy().tobytes()).hexdigest(),
      }
    }
    if tuple(outputs)!=SOURCE_DOMAINS:
        raise ValueError("unknown shifted validation domain")
    if any(not value["sha256"] or value["split"]!="train"
           for value in provenances.values()):
        raise ValueError("invalid source provenance")
    return outputs,provenances
