"""S83 separately AUTHORED new TRAIN-only event-relative area targets.

Constructed labels are NOT independently adjudicated, no supervised
input to original Calibri43M H8 capture, S77 MENTION topK ranking, S82
forward or source-only structural cue builder. Input text can never
import this module. Annotation anchors only select source-only candidate
positions for supervised losses and after-training diagnostics.

Nine distinct textual/genre families each contain 8 differently written
constructions. Last two groups are still synthetic TRAIN diagnostics,
not a one-shot real independent evaluation. Old S79 NOVEL48 is SPENT
and MUST NOT be imported, trained on, rechecked or used for tuning.
"""
from __future__ import annotations

from collections import Counter
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,TRAIN_GROUPS,DIAG_GROUPS,
)

# Fixed constructor-author annotations (event-word, multi-hot factor IDs).
# NOT trustworthy independently adjudicated language annotations.
ANCHORS=(
    (("зафиксировал",()),("отдал",(0,7)),("заверил",(1,)),
     ("оплачен",(2,)),("обработал",(3,6)),("заменил",(4,)),
     ("вручил",(5,6)),("проверил",(6,))),
    (("поднял",()),("открыл",(0,7)),("сообщил",(1,)),
     ("согласован",(2,)),("восстановил",(3,6)),("бросил",(4,)),
     ("передал",(5,6)),("запустил",(6,))),
    (("удостоверил",()),("огласил",(0,7)),("получил",(1,)),
     ("доставлена",(2,)),("изменила",(3,6)),("перечислил",(4,7)),
     ("передал",(5,6)),("описал",(6,))),
    (("отвёз",()),("назначил",(0,7)),("выдал",(1,)),
     ("загружен",(2,)),("восстановила",(3,6)),("отправил",(4,)),
     ("передать",(4,5)),("изучил",(6,))),
    (("опубликовал",()),("пересчитал",(0,7)),("согласовала",(1,)),
     ("подготовлена",(2,)),("восстановил",(3,6)),("одобрил",(4,7)),
     ("направил",(5,6)),("записал",(6,))),
    (("закрепил",()),("доставил",(0,7)),("направлял",(1,)),
     ("вручена",(2,)),("открыл",(3,6)),("поднял",(4,)),
     ("отдал",(5,6)),("выгрузил",(6,))),
    (("развернул",()),("отключил",(0,7)),("установил",(1,)),
     ("закреплён",(2,)),("восстановил",(3,6)),("обработал",(4,)),
     ("переслал",(5,6)),("исправил",(6,))),
    (("собрал",()),("раздал",(0,7)),("внесла",(1,)),
     ("благоустроена",(2,)),("возобновил",(3,6)),("передвинул",(4,)),
     ("передал",(5,6)),("покрасил",(6,))),
    (("остановил",()),("сорвал",(0,7)),("выставляла",(1,)),
     ("утверждён",(2,)),("восстановил",(3,6)),("уронила",(4,)),
     ("передал",(5,6)),("изменил",(6,))),
)
assert len(ANCHORS)==9 and all(len(g)==8 for g in ANCHORS)


def S83_train_loss_gold_only():
    raw=source_only_new72()
    records=[]
    for row,(verb,area_ids) in zip(raw,(u for group in ANCHORS for u in group)):
        words=raw_surface_grammar(row["text"])
        matching=[i for i,z in enumerate(words)
                  if z.form.casefold()==verb.casefold()]
        if not matching or not all(0<=i<len(AREAS) for i in area_ids):
            raise ValueError("S83 manual TRAIN-only event anchor missing or factor OOB")
        if len(set(area_ids))!=len(area_ids):
            raise ValueError("S83 manual TRAIN-only duplicate factor")
        records.append({
            "row_id":row["row_id"],
            "group":row["group"],
            "event_raw_word_index_ONLY_loss_or_posthoc":matching[-1],
            "event_word_source_MATCH_ONLY_loss":verb,
            "eight_area_multi_hot_ONLY_loss":
                tuple(int(i in area_ids) for i in range(len(AREAS))),
            "not_independently_human_adjudicated":True,
        })
    if len(records)!=72:
        raise ValueError("S83 constructed targets incomplete")
    return tuple(records)


def precommit_manifest():
    r=S83_train_loss_gold_only()
    supports=Counter(k for z in r for k,t in zip(AREAS,z["eight_area_multi_hot_ONLY_loss"])
                     if t)
    return {
        "stage":"NRA2_S83_NEW72_DIVERSE_ORIGINALTRAIN_SOURCE_GOLD_HARD_WALL",
        "source_only_sha256":source_only_new72_sha256(),
        "n_source_rows":72,
        "n_lexically_and_style_separated_groups":9,
        "n_train_rows":sum(z["group"] in TRAIN_GROUPS for z in r),
        "n_synthetic_diag_rows":sum(z["group"] in DIAG_GROUPS for z in r),
        "factor_support_author_constructed":dict(supports),
        "event_gold_labels_loaded_only_in_TRAIN_loss_or_posthoc":True,
        "S79_48_one_time_spent_test_NEVER_imported":True,
        "source_only_original43M_H8_FIRST_before_any_labels":True,
        "manually_authored_labels_NOT_independently_adjudicated":True,
        "same_8_factor_names_as_S82":True,
        "real_original43M_native_BPE_not_captured_yet":True,
        "new_S82_neural_parameters_not_trained_yet":True,
        "independent_prospective_new_first_read_NOT_created":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
