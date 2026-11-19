"""NRA2 S76: separate TRAIN-loss-only event-specific construction targets.

S74 raw source and original S75 native H8 forward may NEVER import this.
S74 training labels were utterance-level, and are not universally valid
for the currently focused event. In a sentence 'he gave, but a chief did
NOT send', negation belongs to the second verb, not the first. S76 uses
event-relative TRAIN-only targets to avoid falsely supervising the main
positive event with negation of a different event.

These are template-constructed labels, NOT independently adjudicated human
gold. No hypothesis/output/state/source feature is ever selected by a
gold label: original S65 candidate event slots are computed source-only.
Gold event token index enters ONLY loss and training diagnostics.
"""
from __future__ import annotations

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s74_newtrain120_source_ONLY import (
    FAMILIES,new_source_only_train120,source_sha256,GROUP_TRAIN,GROUP_DIAG,
)
from research.nra3_s20.s72_nra2_situation_microareas import AREAS

SITUATION_EVENT_AREAS={
     1:(),
     2:(0,7),
     3:(1,),
     4:(2,),
     5:(0,7),
     6:(4,5),
     7:(3,6),   # main *second*, successful transfer; first is negated
     8:(6,),    # main positive transfer, negation pertains another event
     9:(5,),
    10:(0,1,7), # quote explicitly denied; event not verified
}
assert len(SITUATION_EVENT_AREAS)==10

def require(x,msg):
    if not bool(x):raise ValueError("S76 S74 TRAIN-only event anchor: "+msg)

def template_constructed_OLDTRAIN_event_area_gold120_for_loss_only():
    rows=new_source_only_train120()
    targets=[]
    for row in rows:
        family=int(row["family"][1:])-1
        variant=int(row["row_id"].split("_")[1])
        require(0<=family<len(FAMILIES)
                and 1<=variant<=10,"frozen source family/variant")
        _,_,_,_,_,_,_,verb,passive=FAMILIES[family]
        anchor=passive if variant==4 else verb
        toks=raw_surface_grammar(row["text"])
        candidates=[i for i,t in enumerate(toks)
                    if t.form.casefold()==anchor.casefold()]
        require(len(candidates)>=1,
                "S76 constructed primary event verb absent")
        # Source case7 explicitly has two identical predicate instances:
        # first negated, second completed. Only loss anchor chooses latter.
        event=candidates[-1] if variant==7 else candidates[0]
        factor=SITUATION_EVENT_AREAS[variant]
        target=tuple(int(i in factor) for i in range(len(AREAS)))
        targets.append({
            "row_id":row["row_id"],
            "family":row["family"],
            "main_event_raw_index_ONLY_in_TRAIN_loss":event,
            "event_area_multi_hot_ONLY_in_TRAIN_loss":target,
            "constructed_not_independent_human_gold":True,
            "not_in_real43m_or_source_only_features":True,
        })
    require(len(targets)==120
            and len({x["row_id"] for x in targets})==120
            and all(len(x["event_area_multi_hot_ONLY_in_TRAIN_loss"])==8
                    and all(a in (0,1) for a in x["event_area_multi_hot_ONLY_in_TRAIN_loss"])
                    for x in targets),
            "S76 exactly 120 NEW TRAIN-only construction anchors")
    return tuple(targets)

def preregistration():
    return {
        "stage":"NRA2_S76_NEWTRAIN120_EVENT_RELATIVE_8AREA_SOURCE_SEPARATE_GOLD",
        "S74_source_sha256":source_sha256(),
        "n_new_source_rows":120,
        "grouped_train_lexical_families":list(GROUP_TRAIN),
        "grouped_old_train_diagnostic_lexical_families":list(GROUP_DIAG),
        "new_event_relative_labels_separate_from_native_forward":True,
        "case7_earlier_negation_not_attributed_to_later_true_event":True,
        "case8_second_clause_negation_not_attributed_to_first_true_event":True,
        "constructed_targets_not_independently_human_verified":True,
        "prospective_independent_real_world_generalization_NOT_claimed":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
