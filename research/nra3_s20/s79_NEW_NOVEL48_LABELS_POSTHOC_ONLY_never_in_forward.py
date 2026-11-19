"""S79 ONE-TIME NOVEL48 CONSTRUCTED gold, physically isolated from H8 & S78.

These 48 manually authored label targets are fixed before first native
Calibri43M/GT/S78 model read. No model source path or original 43M
forward may import this module. Labels are NOT independently human
adjudicated, and do not establish objective natural-language ground
truth. Never use this set to train, calibrate or reselect S78.

Representation: event-relative eight situation attributes in S72 order.
The annotation anchor indexes ONLY a requested source verb for the
teacher-anchored diagnostic, not an input to source-only event proposal,
candidate ranking or S78 classifier forward.
"""
from __future__ import annotations
from collections import Counter
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_only,new_novel48_source_sha256,
)
from research.nra3_s20.s72_nra2_situation_microareas import AREAS

# (event-verb raw surface form, occurrence first=0 last=-1, factor indices)
# All 48 manually fixed, no model-generated annotations or pred inspection.
LABELS=(
    ("занёс",0,()),("выдал",0,()),("перечислила",0,()),
    ("отправил",0,()),("вернула",0,()),("передал",0,()),
    ("заменил",0,(0,7)),("доставил",0,(0,7)),("назначил",0,(0,7)),
    ("открыл",0,(0,7)),("выплатил",0,(0,7)),("выслал",0,(0,7)),
    ("выдал",0,(1,)),("согласовала",0,(1,)),("отправлять",0,(1,)),
    ("назначил",0,(1,)),("подписывала",0,(1,)),("отдал",0,(1,)),
    ("направлена",0,(2,)),("выдан",0,(2,)),("переданы",0,(2,)),
    ("утверждён",0,(2,)),("зарегистрировано",0,(2,)),("проведён",0,(2,)),
    ("заменил",0,(3,6)),("отправил",0,(3,6)),("восстановил",0,(3,6)),
    ("подписал",0,(3,6)),("доставил",0,(3,6)),("заверила",-1,(3,6)),
    ("отправит",0,(4,7)),("вернул",0,(4,)),("заменил",0,(4,)),
    ("установил",0,(4,)),("вручит",0,(4,7)),("разрешил",0,(4,)),
    ("отправил",0,(5,)),("передала",0,(5,)),("выслал",0,(5,)),
    ("отправила",0,(5,7)),("предъявить",0,(4,5)),("протянул",0,(5,7)),
    ("прочистил",0,(6,)),("выдала",0,(6,)),
    ("получил",0,(0,7,6)),("отправил",0,(6,)),
    ("доставил",0,(0,1,7,6)),("утвердил",-1,(3,6)),
)


def frozen_novel48_labels_for_ONE_FINAL_READ_only():
    source=new_novel48_source_only()
    if len(LABELS)!=len(source)!=48:
        raise ValueError("S79 frozen48 literal annotation size")
    result=[]
    for row,(word,occ,factors) in zip(source,LABELS):
        words=raw_surface_grammar(row["text"])
        match=[i for i,z in enumerate(words) if z.form.casefold()==word.casefold()]
        if not match or occ not in (0,-1):
            raise ValueError("S79 anchor absent from new source or invalid occurrence")
        idx=match[occ]
        if len(factors)!=len(set(factors)) or any(
                z<0 or z>=len(AREAS) for z in factors):
            raise ValueError("S79 factor labels outside frozen eight heads")
        result.append({
            "row_id":row["row_id"],
            "constructed_anchor_INDEX_ONLY_FOR_POSTHOC":idx,
            "eight_area_target_POSTHOC":[int(j in factors)
                                         for j in range(len(AREAS))],
            "expected_verb_POSTHOC":word,
            "factors_posthoc_only":[AREAS[i] for i in factors],
            "not_independently_human_adjudicated":True,
        })
    if len(result)!=48:
        raise ValueError("S79 one-time constructed anchors incomplete")
    return tuple(result)


def preregistered_manifest():
    g=frozen_novel48_labels_for_ONE_FINAL_READ_only()
    areas=Counter(name for z in g for name in z["factors_posthoc_only"])
    return {
        "stage":"NRA2_S79_NOVEL48_SYNTHETIC_ONE_TIME_FROZEN_S78_FIRST_READ_PROTOCOL",
        "raw_only_source_sha256":new_novel48_source_sha256(),
        "n_new_syntactic_shift_probes":48,
        "n_distinct_constructed_event_labels":len(g),
        "per_factor_support_AUTHOR_CONSTRUCTED":dict(areas),
        "original_S78_weights_frozen_before_this_source":True,
        "source_gold_forward_wall_required":True,
        "one_time_score_no_new_S78_fitting_or_hparam_selection":True,
        "fixed_eight_situation_heads_no_head_drop":True,
        "fixed_raw_tokenizer_genuine43m_and_first616J7_required":True,
        "manually_constructed_NOT_independently_human_adjudicated":True,
        "novel_syntax_stress_NOT_statistically_representative":True,
        "no_spent_EVAL128_EVAL32_S66_held_gold":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
