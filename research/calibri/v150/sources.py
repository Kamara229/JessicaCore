"""Only fresh SciQ / BoolQ VALID rows 600..623, never official v89 TEST."""
import hashlib
import json
from datasets import load_dataset
from huggingface_hub import HfApi
from v146.sources import Task
from .protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
                       ROW_BEGIN,ROW_END,QUESTION_COUNT,SPLIT,
                       DATASETS,MAX_SUPPORT_CHARS,MAX_PASSAGE_CHARS,check_protocol)


def digest(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def make_sciq(rows):
    entries=[]
    for j,row in enumerate(rows):
        correct=str(row["correct_answer"]).strip()
        distractors=[str(row[k]).strip() for k in ("distractor1","distractor2","distractor3")]
        q=str(row["question"]).strip()
        if not q or not correct or len(set([correct,*distractors]))!=4:
            raise ValueError("incomplete actual SciQ source question")
        answer_slot=(ROW_BEGIN+j)%4
        distractors=distractors[j%3:]+distractors[:j%3]
        answers=distractors.copy()
        answers.insert(answer_slot,correct)
        support=str(row["support"]).strip()
        was_clipped=len(support)>MAX_SUPPORT_CHARS
        support=support[:MAX_SUPPORT_CHARS]
        prompt=("Answer the science question using the choices below. Reply with one letter.\n"+
                (f"Context: {support}\n" if support else "")+
                f"Question: {q}\n"+
                "\n".join(f"{letter}. {choice}" for letter,choice in zip("ABCD",answers))+
                "\nReply with only A, B, C, or D.")
        entries.append(Task("SciQ",ROW_BEGIN+j,prompt,("A","B","C","D"),
                            answer_slot,digest(prompt),was_clipped))
    if len(entries)!=QUESTION_COUNT:raise ValueError("not all original SciQ validation cases")
    return entries


def make_boolq(rows):
    entries=[]
    for j,row in enumerate(rows):
        passage=str(row["passage"]).strip()
        question=str(row["question"]).strip()
        gold=row["answer"]
        if not passage or not question or type(gold) is not bool:
            raise ValueError("invalid BoolQ passage or source bool")
        clipped=len(passage)>MAX_PASSAGE_CHARS
        passage=passage[:MAX_PASSAGE_CHARS]
        prompt=(f"Read the passage and answer the question.\nPassage: {passage}\n"
                f"Question: {question}\nReply with only Yes or No.")
        entries.append(Task("BoolQ",ROW_BEGIN+j,prompt,("Yes","No"),
                            0 if gold else 1,digest(prompt),clipped))
    if len(entries)!=QUESTION_COUNT:raise ValueError("not all original BoolQ validation cases")
    return entries


def load_new_valid_only():
    check_protocol()
    sources={}
    tasks={}
    for key,repo,revision in (("SciQ",SCI_Q,SCI_Q_COMMIT),
                              ("BoolQ",BOOL_Q,BOOL_Q_COMMIT)):
        pinned=HfApi().dataset_info(repo,revision=revision).sha
        if not pinned.lower().startswith(revision.lower()):
            raise ValueError("original registered revision changed")
        # Use the vetted Parquet BoolQ revision, never trust_remote_code.
        rows=load_dataset(repo,revision=pinned,split=f"{SPLIT}[{ROW_BEGIN}:{ROW_END}]")
        if len(rows)!=QUESTION_COUNT:raise ValueError("VALID row size changed")
        tasks[key]=make_sciq(rows) if key=="SciQ" else make_boolq(rows)
        sources[key]={
            "dataset":repo,"revision":pinned,"split":SPLIT,
            "rows":[ROW_BEGIN,ROW_END],
            "exact_question_prompt_SHA256":digest(json.dumps(
                [p.prompt_sha for p in tasks[key]])),
            "supervised_gold_never_used_for_weight_fit":True,
            "context_truncated_count":sum(t.clipped_context for t in tasks[key]),
        }
    if tuple(tasks)!=DATASETS:
        raise ValueError("source order changed")
    return tasks,sources
