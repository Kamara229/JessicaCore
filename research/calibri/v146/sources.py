"""Fetch only two small, pinned Hugging Face VALIDATION slices.

There is NO download of any test split and NO reference to locked v89 data.
Gold labels are not fed as a separate answer in any candidate prompt.
"""
import hashlib,json
from dataclasses import dataclass
from datasets import load_dataset
from huggingface_hub import HfApi

from .protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,SPLIT,START,END,
                       COUNT,MAX_PASSAGE_CHARS,MAX_SUPPORT_CHARS,validate)


@dataclass(frozen=True)
class Task:
    corpus: str
    index: int
    prompt: str
    options: tuple[str,...]
    correct_index: int
    prompt_sha: str
    clipped_context: bool


def _digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sci_questions(records):
    tasks=[]
    for offset,row in enumerate(records):
        q=row["question"].strip()
        gold=row["correct_answer"].strip()
        distractors=[row[k].strip() for k in ("distractor1","distractor2","distractor3")]
        if not q or not gold or len(set([gold]+distractors))!=4:
            raise ValueError("malformed four-choice SciQ question")
        index=(START+offset)%4
        others=distractors[offset%3:]+distractors[:offset%3]
        options=others.copy()
        options.insert(index,gold)
        support=row["support"].strip()
        shortened=len(support)>MAX_SUPPORT_CHARS
        support=support[:MAX_SUPPORT_CHARS]
        prompt=("Answer the science question using the choices below. Reply with one letter.\n"
                +(f"Context: {support}\n" if support else "")
                +f"Question: {q}\n"
                +("\n".join(f"{letter}. {option}" for letter,option
                             in zip("ABCD",options)))
                +"\nReply with only A, B, C, or D.")
        tasks.append(Task("SciQ",START+offset,prompt,("A","B","C","D"),
                          index,_digest(prompt),shortened))
    if len(tasks)!=COUNT:raise ValueError("SciQ lost some validation rows")
    return tasks


def bool_questions(records):
    tasks=[]
    for offset,row in enumerate(records):
        passage=row["passage"].strip()
        question=row["question"].strip()
        if not question or not passage or type(row["answer"]) is not bool:
            raise ValueError("malformed BoolQ reading comprehension record")
        clipped=len(passage)>MAX_PASSAGE_CHARS
        context=passage[:MAX_PASSAGE_CHARS]
        prompt=(f"Read the passage and answer the question.\nPassage: {context}\n"
                f"Question: {question}\nReply with only Yes or No.")
        tasks.append(Task("BoolQ",START+offset,prompt,("Yes","No"),
                          0 if row["answer"] else 1,_digest(prompt),clipped))
    if len(tasks)!=COUNT:raise ValueError("BoolQ lost some validation rows")
    return tasks


def load_valid():
    validate()
    api=HfApi()
    specs=((SCI_Q,SCI_Q_COMMIT),(BOOL_Q,BOOL_Q_COMMIT))
    sources={}
    samples={}
    for dataset_id,commit in specs:
        sha=api.dataset_info(dataset_id,revision=commit).sha
        if not sha.lower().startswith(commit.lower()):
            raise ValueError("dataset revision was not pinned")
        ds=load_dataset(dataset_id,split=f"{SPLIT}[{START}:{END}]",revision=sha)
        if len(ds)!=COUNT:raise ValueError("wrong validation slice")
        identifier="SciQ" if dataset_id==SCI_Q else "BoolQ"
        samples[identifier]=(sci_questions(ds) if identifier=="SciQ"
                              else bool_questions(ds))
        sources[identifier]={
            "dataset":dataset_id,"commit":sha,"split":SPLIT,
            "row_interval":[START,END],
            "validation_prompts_SHA256":_digest(json.dumps(
                [t.prompt_sha for t in samples[identifier]])),
            "gold_labels_NOT_in_model_prompts":True,
            "clipped_passage_count":sum(t.clipped_context for t in samples[identifier]),
        }
    return samples,sources
