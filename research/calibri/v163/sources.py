"""Original public question VALIDATION only. Never protected v89 TEST."""
import hashlib,json
from datasets import load_dataset
from huggingface_hub import HfApi
from v146.sources import Task
from .protocol import (SCI_Q,SCI_Q_COMMIT,BOOL_Q,BOOL_Q_COMMIT,
    MAX_SUPPORT_CHARS,MAX_PASSAGE_CHARS,DATASETS,
    ROW_BEGIN,ROW_END,COUNT,SPLIT,check)


def digest(value):
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def questions_sciq(rows):
    results=[]
    for j,row in enumerate(rows):
        current=ROW_BEGIN+j
        correct=str(row["correct_answer"]).strip()
        wrong=[str(row[field]).strip()
               for field in ("distractor1","distractor2","distractor3")]
        if not correct or any(not x for x in wrong) or len(set((correct,*wrong)))!=4:
            raise ValueError("duplicate or empty original SciQ candidate answers")
        desired_position=current%4
        wrong=wrong[j%3:]+wrong[:j%3]
        options=list(wrong)
        options.insert(desired_position,correct)
        context=str(row["support"]).strip()
        clipped=len(context)>MAX_SUPPORT_CHARS
        question=str(row["question"]).strip()
        if not question:raise ValueError("empty original SciQ question")
        prompt=("Answer this science question. Reply with one letter.\n"+
                (f"Context: {context[:MAX_SUPPORT_CHARS]}\n" if context else "")+
                f"Question: {question}\n"+
                "\n".join(f"{letter}. {text}" for letter,text in zip("ABCD",options))+
                "\nReply with only A, B, C, or D.")
        results.append(Task("SciQ",current,prompt,("A","B","C","D"),
                            desired_position,digest(prompt),clipped))
    return results


def questions_boolq(rows):
    results=[]
    for j,row in enumerate(rows):
        question=str(row["question"]).strip()
        passage=str(row["passage"]).strip()
        answer=row["answer"]
        if not question or not passage or type(answer) is not bool:
            raise ValueError("missing BoolQ question, passage or boolean label")
        clipped=len(passage)>MAX_PASSAGE_CHARS
        prompt=(f"Read the passage and answer the question.\n"
                f"Passage: {passage[:MAX_PASSAGE_CHARS]}\n"
                f"Question: {question}\nReply with only Yes or No.")
        results.append(Task("BoolQ",ROW_BEGIN+j,prompt,("Yes","No"),
                            0 if answer else 1,digest(prompt),clipped))
    return results


def load_frozen_validation():
    check()
    tasks={}
    origins={}
    for domain,repo,revision in (
        ("SciQ",SCI_Q,SCI_Q_COMMIT),
        ("BoolQ",BOOL_Q,BOOL_Q_COMMIT),
    ):
        sha=HfApi().dataset_info(repo,revision=revision).sha
        if not sha.lower().startswith(revision.lower()):
            raise ValueError("original approved original QA commit changed")
        records=load_dataset(repo,revision=sha,
                             split=f"{SPLIT}[{ROW_BEGIN}:{ROW_END}]")
        if len(records)!=COUNT:
            raise ValueError("precommitted number of original questions differs")
        cases=(questions_sciq(records) if domain=="SciQ"
               else questions_boolq(records))
        if len(cases)!=COUNT or any(x.index!=ROW_BEGIN+i for i,x in enumerate(cases)):
            raise ValueError("missing new science/boolean question")
        tasks[domain]=cases
        origins[domain]={
            "original_dataset":repo,"revision":sha,
            "original_split":SPLIT,"original_row_range":[ROW_BEGIN,ROW_END],
            "prompt_sha_sequence_SHA256":digest(json.dumps([x.prompt_sha for x in cases])),
            "context_clipped_count":sum(x.clipped_context for x in cases),
            "no_answers_in_weight_fit_or_extra_source_training":True,
            "official_v89_independent_TEST_unopened":True,
        }
    if tuple(tasks)!=DATASETS:raise ValueError("original QA domain changed")
    return tasks,origins
