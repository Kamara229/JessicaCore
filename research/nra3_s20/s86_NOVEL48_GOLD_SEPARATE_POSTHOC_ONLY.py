"""NRA2 S86: SEPARATE one-shot synthetic first-read annotations ONLY.

NEVER import in original Calibri43M+GT native BPE, S87 H8 source
capture, original S59 source-only mention ranks, or frozen S85/S82
forward. Model outputs MUST be physically persisted BEFORE this
annotation module is imported. Target author is assistant, not an
independent human adjudicator, so neither synthetic QA nor natural
language objective gold is claimed. These 48 become permanently SPENT
after S88 first inference. No checkpoint/hparam/threshold tuning.
"""
from __future__ import annotations
from collections import Counter

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS
from research.nra3_s20.s86_newSYNTHETIC48_SOURCEONLY_never_train import (
    S86_novel48_source_only,S86_novel48_source_sha256,
)

# Eight groups of six authored cases, each fixed event lexical token and
# nonexclusive event-relative factor IDs, not whole sentence semantics.
# idx0 quote/report, idx1 local negation, idx2 passive, idx3 temporal
# revision, idx4 conditional/dream, idx5 coreference, idx6 multiple
# competing propositions, idx7 uncertainty.
AUTHOR_ANCHORS=(
    ("передал",(0,7)),("утвердил",(0,7)),("сбил",(0,7)),
    ("отключил",(0,7)),("передал",(0,7)),("отменил",(0,7)),
    ("перечислил",(1,)),("отправила",(1,)),("зарегистрировал",(1,)),
    ("поставила",(1,)),("доставил",(1,)),("выдала",(1,)),
    ("оформлен",(2,)),("промаркированы",(2,)),("внесены",(2,)),
    ("заверена",(2,)),("составлен",(2,)),("изготовлен",(2,)),
    ("перезапустил",(3,6)),("подписал",(3,6)),("открыл",(3,6)),
    ("назначил",(3,6)),("отправил",(3,6)),("поставил",(3,6)),
    ("завершит",(4,7)),("вышел",(4,7)),("потерял",(4,)),
    ("изменил",(4,)),("назначит",(4,7)),("вручает",(4,)),
    ("переслала",(5,6)),("передал",(5,7)),("показал",(5,6)),
    ("выписал",(5,6)),("объяснил",(5,6)),("вернула",(4,5)),
    ("исправил",(6,)),("привёз",(6,)),("одобрил",(6,)),
    ("поменял",(6,)),("покрывал",(6,)),("согласовал",(6,)),
    ("направил",(7,)),("оплатил",(7,)),("восстановил",(7,)),
    ("передать",(7,)),("перепутала",(7,)),("выслал",(4,7)),
)
assert len(AUTHOR_ANCHORS)==48


def S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only():
    sources=S86_novel48_source_only()
    if len(sources)!=48 or len(AUTHOR_ANCHORS)!=48:
        raise ValueError("S86 preregistered source and constructed targets must be 48")
    rec=[]
    for row,(verb,areas) in zip(sources,AUTHOR_ANCHORS):
        toks=raw_surface_grammar(row["text"])
        match=[i for i,w in enumerate(toks)
               if w.form.casefold()==verb.casefold()]
        if len(match)!=1:
            raise ValueError(f"S86 first-read anchor {row['row_id']} {verb!r} found {len(match)}")
        if len(set(areas))!=len(areas) or any(not 0<=z<len(AREAS) for z in areas):
            raise ValueError("S86 event-relative factor labels not in frozen eight")
        rec.append({
            "row_id":row["row_id"],
            "original_event_word_index_POSTHOC_ONLY":match[0],
            "original_event_word_form_POSTHOC_ONLY":verb,
            "event_situation_factor8_POSTHOC_ONLY":
                [int(k in areas) for k in range(len(AREAS))],
            "independently_human_adjudicated":False,
        })
    return tuple(rec)


def preregistered_S86_firstread_manifest():
    g=S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only()
    supports=Counter(j for r in g for j,v in
                     enumerate(r["event_situation_factor8_POSTHOC_ONLY"]) if v)
    return {
        "stage":"NRA2_S86_NOVEL48_SOURCE_GOLD_PRECOMMIT_S85_FIRST_READ",
        "new_source_only_SHA256":S86_novel48_source_sha256(),
        "n_new_syntactic_firstread_cases":48,
        "n_author_locked_verb_and_eightfactor_targets":len(g),
        "precommitted_factor_support_INDEX":dict(supports),
        "original_S85_8264_pretrained_FP16_weights_must_BE_frozen_before_firstread":True,
        "new_sources_NOT_in_old_S83_new72_training_or_diag":True,
        "no_old_S79_NOVEL48_retrain_or_rescore":True,
        "first_genuine43M_H8_only_SOURCE_before_labels":True,
        "source_only_S59_mention_rank_before_labels":True,
        "S85_source_only_logits_PHYSICALLY_SAVED_BEFORE_gold_open":True,
        "synthetic_author_labels_not_independent_human_truth":True,
        "no_checkpoint_selection_no_threshold_search_no_training_after_firstread":True,
        "original_asserted_S65_reported_quote_mask_must_remain":True,
        "event_top1_and_top8_are_independent_of_gold":True,
        "original_616_baseline_protected_candidate_J7_not_tested":True,
        "online_causal_and_all_in_bpp_CPU_not_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
