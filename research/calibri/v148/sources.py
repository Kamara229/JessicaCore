"""Fresh, immutable VALIDATION-only SciQ/BoolQ answer-label benchmark.

Fixed 16 questions each at row indexes [400,416), distinct from v146
rows [200,208); no official v89 TEST and no data used for training.
"""
import hashlib,json
from datasets import load_dataset
from huggingface_hub import HfApi
from v146.sources import Task
from .protocol import (DOMAINS,SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
                       START,END,COUNT,SPLIT,MAX_SUPPORT_CHARS,MAX_PASSAGE_CHARS,
                       verify_protocol)


def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_sciq(records):
    tasks=[]
    for k,r in enumerate(records):
        question=str(r["question"]).strip()
        good=str(r["correct_answer"]).strip()
        wrong=[str(r[t]).strip() for t in ("distractor1","distractor2","distractor3")]
        if not question or not good or len(set([good,*wrong]))!=4:
            raise ValueError("incomplete SciQ option source")
        # Permute distractors independently of the answer; gold enters options
        # but NOT as privileged separate model-facing reference label.
        wrong=wrong[k%3:]+wrong[:k%3]
        chosen=(START+k)%4
        answers=wrong.copy()
        answers.insert(chosen,good)
        context=str(r["support"]).strip()
        clipped=len(context)>MAX_SUPPORT_CHARS
        context=context[:MAX_SUPPORT_CHARS]
        prompt=("Answer the science question using the choices below. Reply with one letter.\n"
                +(f"Context: {context}\n" if context else "")
                +f"Question: {question}\n"
                +"\n".join(f"{letter}. {option}" for letter,option in zip("ABCD",answers))
                +"\nReply with only A, B, C, or D.")
        tasks.append(Task("SciQ",START+k,prompt,("A","B","C","D"),
                          chosen,digest(prompt),clipped))
    if len(tasks)!=COUNT:raise ValueError("incomplete new SciQ VALID")
    return tasks


def make_boolq(records):
    tasks=[]
    for k,r in enumerate(records):
        passage=str(r["passage"]).strip()
        q=str(r["question"]).strip()
        if not passage or not q or type(r["answer"]) is not bool:
            raise ValueError("bad BoolQ source passage or reference")
        clipped=len(passage)>MAX_PASSAGE_CHARS
        passage=passage[:MAX_PASSAGE_CHARS]
        prompt=(f"Read the passage and answer the question.\nPassage: {passage}\n"
                f"Question: {q}\nReply with only Yes or No.")
        tasks.append(Task("BoolQ",START+k,prompt,("Yes","No"),
                          0 if r["answer"] else 1,digest(prompt),clipped))
    if len(tasks)!=COUNT:raise ValueError("incomplete new BoolQ VALID")
    return tasks


def load_fresh_validation():
    verify_protocol()
    sources={}
    tasks={}
    for key,dataset,revision in (("SciQ",SCI_Q,SCI_Q_COMMIT),
                                 ("BoolQ",BOOL_Q,BOOL_Q_COMMIT)):
        sha=HfApi().dataset_info(dataset,revision=revision).sha
        if not sha.lower().startswith(revision.lower()):
            raise ValueError("source revision changed")
        # BoolQ pinned Parquet conversion commit: no trust_remote_code.
        # load_dataset may download metadata for other splits, but ONLY
        # the requested validation rows are scored and no test is used.
        rows=load_dataset(dataset,revision=sha,split=f"{SPLIT}[{START}:{END}]")
        if len(rows)!=COUNT:raise ValueError("fresh slice size mismatch")
        tasks[key]=make_sciq(rows) if key=="SciQ" else make_boolq(rows)
        sources[key]={
            "dataset":dataset,"revision":sha,"split":SPLIT,
            "range":[START,END],
            "prompt_hashes_sha256":digest(json.dumps([t.prompt_sha for t in tasks[key]])),
            "gold_used_only_for_scoring":True,
            "clipped_contexts":sum(t.clipped_context for t in tasks[key]),
        }
    if tuple(tasks)!=DOMAINS:raise ValueError("incorrect domain ordering")
    return tasks,sources
