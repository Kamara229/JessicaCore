"""NRA2 3.0 S8: TRAIN-only causal weak targets for shared 3-way router.

Labels are a bounded *training heuristic* built solely from the immutable
twelve Russian UD TRAIN rows. They are NOT actual evidence that a specific
branch is correct, not evaluation truth, and NEVER available at inference.

At the current token, do not select a future dependency head. A local
backward dependency, a preceding complementarily-typed entity, and a
globally unresolved dependency are three distinguishable target families.
"""
from __future__ import annotations

import torch

from research.nra3_s4.complete_prefix_training_contract import CausalAlignment
from research.nra3_s4.russian_ud_train_only import (
    RussianTrainSentence,TRAIN_SPLIT,build_train,
)
from research.nra3_s7.parallel_hypotheses import _ROLE_EDGES


GLOBAL=0
LOCAL=1
ROLE=2
BRANCHES=("global_causal","local_recent_causal",
          "model_predicted_semantic_complementarity")
_PAIR=frozenset(tuple(sorted((a,b))) for a,b in _ROLE_EDGES)


def train_only_causal_branch_targets(
        row:RussianTrainSentence, aligned:CausalAlignment,
        *,local_window:int=4)->torch.Tensor:
    """One heuristic branch target per completed UD word, with no label leak.

    Targets are calculated ONLY in the optimizer's loss and MUST NOT enter
    NRA3 forward/inference. This function has no access to EVAL or answers.
    """
    source={r.sha256() for r in build_train()}
    if (not isinstance(row,RussianTrainSentence) or
        row.split!=TRAIN_SPLIT or row.sha256() not in source or
        not isinstance(aligned,CausalAlignment) or
        aligned.independent_eval_labels_loaded or
        aligned.original_prompt_sha256!=row.sha256() or
        len(row.words)!=len(row.roles)!=len(aligned.word_alignment) or
        not 1<=local_window<=4 or
        aligned.complete_prefix_position!=aligned.tokenizer_token_count-1):
        raise ValueError("S8 branch supervision rejects unregistered TRAIN/offsets")
    pos=[w.last_token for w in aligned.word_alignment]
    if (not all(type(p) is int and 0<=p<aligned.tokenizer_token_count
                for p in pos) or
        not all(pos[i]<pos[i+1] for i in range(len(pos)-1))):
        raise ValueError("S8 cannot use invalid or future wordpiece positions")

    targets=[]
    for i,(word,role) in enumerate(zip(row.words,row.roles)):
        # Source-bound complements are causal: gold roles only for earlier
        # words, never the current word's future counterpart. This is a
        # TRAIN-only weak teacher, not a hard-coded inference answer.
        complementary=any(tuple(sorted((role,old_role))) in _PAIR
                          for old_role in row.roles[:i])
        head=word.head
        observed=(head>0 and head<=len(row.words)
                  and pos[head-1]<=pos[i])
        nearby=observed and pos[i]-pos[head-1]<local_window
        if role!="NONE" and complementary:
            choice=ROLE
        elif nearby:
            choice=LOCAL
        else:
            choice=GLOBAL
        targets.append(choice)
    return torch.tensor(targets,dtype=torch.long)


def locked_train12_weak_target_counts()->dict[str,int]:
    """Diagnostics only. These are not independent test labels."""
    from research.nra3_s4.complete_prefix_training_contract import (
        AlignedWord,
    )
    counts={name:0 for name in BRANCHES}
    for row in build_train():
        # Token positions preserve UD order, not real Calibri offsets.
        # This only checks *label class existence*, never TRAIN fitting.
        words=tuple(AlignedWord(i+1,(i,),i)
                    for i in range(len(row.words)))
        al=CausalAlignment(row.sha256(),len(words),tuple(range(len(words))),
                           words,len(words)-1)
        targets=train_only_causal_branch_targets(row,al)
        for k in targets.tolist():
            counts[BRANCHES[k]]+=1
    if min(counts.values())<2:
        raise ValueError("S8 weak TRAIN supervision requires 3 populated classes")
    return counts
