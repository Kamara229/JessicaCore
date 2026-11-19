"""NRA2 S74: SEPARATE source-template-constructed NEW TRAIN-only situation labels.

Never import into any original Calibri43M tokenizer, H8 hook, cached
source capture, source-only router or inference call. For subsequent
offline supervision, derive labels solely inside the training LOSS.
These are construction-derived NOT independently human-reviewed labels,
not naturalistic or evaluated cases. Fixed 12 families×10 variants.
"""
from __future__ import annotations

from collections import Counter

from research.nra3_s20.s74_newtrain120_source_ONLY import (
    new_source_only_train120,source_sha256,
)
AREAS=(
    "quoted_or_reported",
    "negation_scope",
    "passive_role_assignment",
    "temporal_revision",
    "hypothetical_or_dream",
    "reference_resolution",
    "competing_event_clauses",
    "uncertainty_abstain",
)
# Event links are predicate+recipient "actually done" or "explicitly rejected";
# NONE for quote/rumor/dream without independently asserted transfer.
CONTRASTS=(
    ("ASSERTED","DIRECTED_ASSERTED",()),
    ("REPORTED_UNCONFIRMED","NONE",(0,7)),
    ("NEGATED","DIRECTED_REJECTED",(1,)),
    ("PASSIVE_ASSERTED","DIRECTED_ASSERTED",(2,)),
    ("UNVERIFIED_REPORT","NONE",(0,7)),
    ("DREAM_UNREALIZED","NONE",(4,5)),
    ("TEMPORAL_FAILED_THEN_ASSERTED","DIRECTED_ASSERTED",(1,3,6)),
    ("COMPETING_CLAUSES","DIRECTED_ASSERTED",(1,6)),
    ("COREFERENCE_ASSERTED","DIRECTED_ASSERTED",(5,)),
    ("DENIED_QUOTATION","NONE",(0,1,7)),
)
FACTUALITY=frozenset(x[0] for x in CONTRASTS)
LINK=frozenset(("NONE","DIRECTED_ASSERTED","DIRECTED_REJECTED"))


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA2 S74 separate TRAIN-only loss labels: "+why)


def constructed_NEW_TRAIN_gold120_only_for_loss():
    raw=new_source_only_train120()
    labels=[]
    for row in raw:
        variant=int(row["row_id"].split("_")[1])
        require(1<=variant<=len(CONTRASTS),"S74 fixed situation variant ABI")
        factuality,link,areas=CONTRASTS[variant-1]
        label=tuple(int(i in areas) for i in range(len(AREAS)))
        labels.append({
            "row_id":row["row_id"],
            "family":row["family"],
            "factuality":factuality,
            "link":link,
            "area_multi_hot":label,
            "area_indices":areas,
            "actually_transferred":link=="DIRECTED_ASSERTED",
            "asserted_directed_recipient":
                link=="DIRECTED_ASSERTED",
            "rejected_directed_recipient":
                link=="DIRECTED_REJECTED",
            "human_independent_gold_verified":False,
            "source_sha256_before_any_native_forward":source_sha256(),
            "TRAIN_only_never_in_source_features":True,
        })
    require(len(labels)==120
            and all(len(x["area_multi_hot"])==len(AREAS)
                    and all(z in (0,1) for z in x["area_multi_hot"])
                    and x["factuality"] in FACTUALITY and x["link"] in LINK
                    and not x["human_independent_gold_verified"]
                    for x in labels),
            "S74 120 construction-derived labels complete")
    # All ten situations must be represented EQUALLY in each lexical family;
    # quote / negation / passives / time / coreference have overlaps.
    histogram=Counter(x["factuality"] for x in labels)
    require(set(histogram)==FACTUALITY and set(histogram.values())=={12},
            "S74 factors balanced across old lexical families")
    require(any(sum(x["area_multi_hot"])>=3 for x in labels),
            "S74 at least one multi-area overlap")
    return tuple(labels)


def new_train120_source_gold_alignment_manifest():
    x=new_source_only_train120()
    g=constructed_NEW_TRAIN_gold120_only_for_loss()
    require([(z["row_id"],z["family"]) for z in x]
            ==[(z["row_id"],z["family"]) for z in g],
            "TRAIN source label IDs and grouped folds unchanged")
    return {
        "stage":"NRA2_S74_NEW_TRAIN120_CONSTRUCTION_DERIVED_SITUATION_MULTI_HOT",
        "source_sha256":source_sha256(),
        "new120_predeclared_before_any_original43M_cpu_native_capture":True,
        "NRA3_S66_SPENT32_NOT_IMPORTED":True,
        "old_spent_EVAL32_EVAL128_NOT_IMPORTED":True,
        "entire_source_gold_split_train_only":True,
        "h8_tokenizer_source_rows_contain_only_id_family_text":True,
        "n_source_rows":len(x),
        "n_construction_derived_labels":len(g),
        "n_unique_lexical_participant_families":12,
        "n_situation_variants_per_family":len(CONTRASTS),
        "labels_not_independently_human_reviewed":True,
        "prospective_first_read_EVAL_NOT_CREATED":True,
        "standalone_true43m_GT_native_CPU_H8_NOT_CAPTURED":True,
        "trained_new_microareas_NOT_EXECUTED":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
