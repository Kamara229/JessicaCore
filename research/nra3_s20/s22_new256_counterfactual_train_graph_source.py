"""NRA3 S22: 256 *synthetic TRAIN-only* counterfactual event-graph sentences.

Eight new lexical families x 32 source-designed surface constructions. This
corrects the S21 zero-exposure COORDINATION construction fold and supplies
multiple independent construction variants for all four event relations and
REJECTED argument scope. Two minimal contrast pairs ('если' vs 'когда')
change event semantics with the rest of the surface sequence held constant.

These are graph supervision examples, NEVER oracle inputs or independent
natural-Russian EVAL. Physical true43M+Green Tunnel CPU capture on these 256
inputs must happen in a separate later stage before any optimizer is run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from research.nra3_s19.graph_schema import canonical_train_only_sha
from research.nra3_s19.train_only_event_graph_source import build_s19_train_only
from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    _arg, _row, build_s20_synthetic_train96,
)

STAGE = "NRA3_S22_NEW256_SYNTHETIC_COUNTERFACTUAL_TRAIN_ONLY"
EXPECTED = 256
FAMILY_COUNT = 8
PATTERN_COUNT = 32
# Filled only after source-generation CI reports the real canonical SHA.
# A follow-up commit must pin it and rerun CI before downstream use.
FROZEN_S22_TRAIN256_SHA256 = "8963e2e8758b1c6a6fbd28eb597a77be1a454d732d0ecac93df6f5a753f6a361"

@dataclass(frozen=True)
class Lexicon:
    male: str
    female: str
    object: str
    male_past: str
    male_future: str
    female_past: str
    female_future: str
    recipient: str
    alternative: str


FAMILIES = (
    Lexicon("Егор", "Вера", "заказ", "оформил", "оформит", "передала", "передаст", "клиенту", "начальнику"),
    Lexicon("Кирилл", "Зоя", "пакет", "собрал", "соберёт", "отправила", "отправит", "курьеру", "бригадиру"),
    Lexicon("Глеб", "Лидия", "отчёт", "подготовил", "подготовит", "передала", "передаст", "редактору", "директору"),
    Lexicon("Борис", "Алина", "список", "составил", "составит", "показала", "покажет", "инспектору", "руководителю"),
    Lexicon("Денис", "Ксения", "запрос", "зарегистрировал", "зарегистрирует", "направила", "направит", "оператору", "инженеру"),
    Lexicon("Лев", "Дарья", "макет", "создал", "создаст", "показала", "покажет", "дизайнеру", "заказчику"),
    Lexicon("Артур", "Нина", "пропуск", "оформил", "оформит", "передала", "передаст", "охраннику", "дежурному"),
    Lexicon("Фёдор", "Полина", "чек", "подписал", "подпишет", "отправила", "отправит", "кассиру", "бухгалтеру"),
)

PATTERNS = (
    "coord_and", "coord_semicolon_reverse", "coord_additionally",
    "coord_a_reverse", "coord_too", "coord_and_more",
    "conditional_if_future", "conditional_result_first",
    "conditional_provided_that", "conditional_only_if_reversed",
    "conditional_subjunctive_first", "conditional_subjunctive_result_first",
    "temporal_when_future", "temporal_when_future_result_first",
    "temporal_first_then", "temporal_after_that_result_first",
    "temporal_as_soon_as", "temporal_before_that",
    "contrast_but", "contrast_although_result_first",
    "contrast_despite_that", "contrast_although_reversed",
    "contrast_nevertheless", "contrast_instead",
    "rejected_not_recipient_but", "rejected_recipient_preposed",
    "rejected_positive_first", "rejected_object_fronted",
    "negated_female_past", "negated_male_past",
    "future_male", "future_female",
)

RELATION_OF = {
    **{p: "COORDINATION" for p in PATTERNS[0:6]},
    **{p: "CONDITION_RESULT" for p in PATTERNS[6:12]},
    **{p: "TEMPORAL_SEQUENCE" for p in PATTERNS[12:18]},
    **{p: "CONTRAST" for p in PATTERNS[18:24]},
}

# Pair indices: ONLY the connecting word changes ("Если" -> "Когда").
COUNTERFACTUAL_PAIRS = ((6, 12), (7, 13))

def _paired(f, kind, pattern):
    a, b, obj = f.male, f.female, f.object
    ap, af, bp, bf = f.male_past, f.male_future, f.female_past, f.female_future
    r, alt = f.recipient, f.alternative

    agent_m = (_arg(1, a, "AGENT"), _arg(1, obj, "PATIENT"))
    agent_f_pron = (_arg(2, b, "AGENT"), _arg(2, "его", "PATIENT"))
    agent_f = (_arg(1, b, "AGENT"), _arg(1, obj, "PATIENT"))
    agent_m_pron = (_arg(2, a, "AGENT"), _arg(2, "его", "PATIENT"))

    def two(text, heads, arguments, relation, direction=(1,2)):
        return _row(text, heads, arguments, refs=(("его",obj),),
                    links=((direction[0],direction[1],relation),))

    if pattern == "coord_and":
        return two("{} {} {}, и {} {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron, "COORDINATION")
    if pattern == "coord_semicolon_reverse":
        return two("{} {} {}; {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron, "COORDINATION")
    if pattern == "coord_additionally":
        return two("{} {} {}. Кроме того, {} {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron,"COORDINATION")
    if pattern == "coord_a_reverse":
        return two("{} {} {}, а {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"COORDINATION")
    if pattern == "coord_too":
        return two("{} {} {}; {} тоже {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron,"COORDINATION")
    if pattern == "coord_and_more":
        return two("{} {} {} и, вдобавок, {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"COORDINATION")

    # If/when minimal pairs have identical event/argument surfaces,
    # but differ in future factuality and directed inter-event relation.
    if pattern in ("conditional_if_future","temporal_when_future"):
        is_if = pattern.startswith("conditional")
        clause = "Если" if is_if else "Когда"
        tag = "CONDITIONAL" if is_if else "FUTURE_UNREALIZED"
        rel = "CONDITION_RESULT" if is_if else "TEMPORAL_SEQUENCE"
        return two("{} {} {} {}, {} {} его.".format(clause,a,af,obj,b,bf),
                   ((af,tag),(bf,tag)),agent_m+agent_f_pron,rel)
    if pattern in ("conditional_result_first","temporal_when_future_result_first"):
        is_if = pattern.startswith("conditional")
        clause = "если" if is_if else "когда"
        tag = "CONDITIONAL" if is_if else "FUTURE_UNREALIZED"
        rel = "CONDITION_RESULT" if is_if else "TEMPORAL_SEQUENCE"
        return two("{} {} {}, {} {} {} его.".format(b,bf,obj,clause,a,af),
                   ((bf,tag),(af,tag)),agent_f+agent_m_pron,rel,(2,1))
    if pattern == "conditional_provided_that":
        return two("При условии, что {} {} {}, {} {} его.".format(a,af,obj,b,bf),
                   ((af,"CONDITIONAL"),(bf,"CONDITIONAL")),
                   agent_m+agent_f_pron,"CONDITION_RESULT")
    if pattern == "conditional_only_if_reversed":
        return two("Только если {} {} {}, {} {} его.".format(b,bf,obj,a,af),
                   ((bf,"CONDITIONAL"),(af,"CONDITIONAL")),
                   agent_f+agent_m_pron,"CONDITION_RESULT")
    if pattern == "conditional_subjunctive_first":
        return two("Если бы {} {} {}, {} {} бы его.".format(a,ap,obj,b,bp),
                   ((ap,"CONDITIONAL"),(bp,"CONDITIONAL")),
                   agent_m+agent_f_pron,"CONDITION_RESULT")
    if pattern == "conditional_subjunctive_result_first":
        return two("{} {} бы {}, если бы {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"CONDITIONAL"),(ap,"CONDITIONAL")),
                   agent_f+agent_m_pron,"CONDITION_RESULT",(2,1))
    if pattern == "temporal_first_then":
        return two("Сначала {} {} {}; затем {} {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron,"TEMPORAL_SEQUENCE")
    if pattern == "temporal_after_that_result_first":
        return two("{} {} {} после того, как {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"TEMPORAL_SEQUENCE",(2,1))
    if pattern == "temporal_as_soon_as":
        return two("Как только {} {} {}, {} {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron,"TEMPORAL_SEQUENCE")
    if pattern == "temporal_before_that":
        return two("До того как {} {} {}, {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"TEMPORAL_SEQUENCE",(2,1))

    if pattern == "contrast_but":
        return two("{} {} {}, но {} не {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"NEGATED")),
                   agent_m+agent_f_pron,"CONTRAST")
    if pattern == "contrast_although_result_first":
        return two("{} не {} {}, хотя {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"NEGATED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"CONTRAST",(2,1))
    if pattern == "contrast_despite_that":
        return two("Несмотря на то что {} {} {}, {} не {} его.".format(a,ap,obj,b,bp),
                   ((ap,"ASSERTED"),(bp,"NEGATED")),
                   agent_m+agent_f_pron,"CONTRAST")
    if pattern == "contrast_although_reversed":
        return two("{} {} {}, хотя {} не {} его.".format(b,bp,obj,a,ap),
                   ((bp,"ASSERTED"),(ap,"NEGATED")),
                   agent_f+agent_m_pron,"CONTRAST",(2,1))
    if pattern == "contrast_nevertheless":
        return two("{} не {} {}; тем не менее, {} {} его.".format(a,ap,obj,b,bp),
                   ((ap,"NEGATED"),(bp,"ASSERTED")),
                   agent_m+agent_f_pron,"CONTRAST")
    if pattern == "contrast_instead":
        return two("{} не {} {}, зато {} {} его.".format(b,bp,obj,a,ap),
                   ((bp,"NEGATED"),(ap,"ASSERTED")),
                   agent_f+agent_m_pron,"CONTRAST")

    # NEGATED event and REJECTED participant are separate labels.
    rej_args=(_arg(1,b,"AGENT"),_arg(1,obj,"PATIENT"),
              _arg(1,r,"RECIPIENT","REJECTED"),_arg(1,alt,"RECIPIENT"))
    if pattern == "rejected_not_recipient_but":
        return _row("{} {} {} не {}, а {}.".format(b,bp,obj,r,alt),
                    ((bp,"ASSERTED"),),rej_args)
    if pattern == "rejected_recipient_preposed":
        return _row("Не {}, а {} {} {} {}.".format(r,alt,b,bp,obj),
                    ((bp,"ASSERTED"),),rej_args)
    if pattern == "rejected_positive_first":
        return _row("{} {} {} {}, но не {}.".format(b,bp,obj,alt,r),
                    ((bp,"ASSERTED"),),rej_args)
    if pattern == "rejected_object_fronted":
        return _row("{} {} {} не {}, а {}.".format(obj.capitalize(),b,bp,r,alt),
                    ((bp,"ASSERTED"),),
                    (_arg(1,b,"AGENT"),_arg(1,obj.capitalize(),"PATIENT"),
                     _arg(1,r,"RECIPIENT","REJECTED"),_arg(1,alt,"RECIPIENT")))

    if pattern == "negated_female_past":
        return _row("Сегодня {} не {} {} {}.".format(b,bp,obj,r),
                    ((bp,"NEGATED"),),
                    (_arg(1,b,"AGENT"),_arg(1,obj,"PATIENT"),_arg(1,r,"RECIPIENT")))
    if pattern == "negated_male_past":
        return _row("{} так и не {} {}.".format(a,ap,obj),
                    ((ap,"NEGATED"),),
                    (_arg(1,a,"AGENT"),_arg(1,obj,"PATIENT")))
    if pattern == "future_male":
        return _row("Завтра {} {} {}.".format(a,af,obj),
                    ((af,"FUTURE_UNREALIZED"),),
                    (_arg(1,a,"AGENT"),_arg(1,obj,"PATIENT")))
    if pattern == "future_female":
        return _row("На следующей неделе {} {} {} {}.".format(b,bf,obj,r),
                    ((bf,"FUTURE_UNREALIZED"),),
                    (_arg(1,b,"AGENT"),_arg(1,obj,"PATIENT"),_arg(1,r,"RECIPIENT")))
    raise ValueError("unregistered semantic source TRAIN construction: "+str(pattern))


def build_s22_train256():
    rows = tuple(_paired(f, "TRAIN_ONLY", pat) for f in FAMILIES for pat in PATTERNS)
    if len(rows) != EXPECTED or len({r.text for r in rows}) != EXPECTED:
        raise ValueError("S22 duplicate/new256 wrong size")
    prior = {g.text for g in build_s19_train_only()} | {
        g.text for g in build_s20_synthetic_train96()}
    if prior.intersection(r.text for r in rows):
        raise ValueError("S22 duplicates prior S19 or S20 source")
    return rows


def splits():
    """No identical lexical family or syntax construction leaks across fold."""
    out = []
    for fam in range(FAMILY_COUNT):
        held = tuple(i for i in range(EXPECTED) if i // PATTERN_COUNT == fam)
        tr = tuple(i for i in range(EXPECTED) if i // PATTERN_COUNT != fam)
        out.append(("LEXICAL", fam, tr, held))
    for pat in range(PATTERN_COUNT):
        held = tuple(i for i in range(EXPECTED) if i % PATTERN_COUNT == pat)
        tr = tuple(i for i in range(EXPECTED) if i % PATTERN_COUNT != pat)
        out.append(("CONSTRUCTION", pat, tr, held))
    for name, group, train, held in out:
        if set(train)&set(held) or len(train)+len(held) != EXPECTED:
            raise ValueError("S22 predeclared TRAIN-only fold leak")
    return tuple(out)


def verify_coverage(rows=None):
    rows = build_s22_train256() if rows is None else rows
    rel = Counter(link.relation for row in rows for link in row.event_relations)
    rejection = sum(a.polarity == "REJECTED" for g in rows for a in g.arguments)
    assert rel == {"COORDINATION":48,"CONDITION_RESULT":48,
                   "TEMPORAL_SEQUENCE":48,"CONTRAST":48}
    assert rejection == 32
    for axis,group,train,held in splits():
        trrel=Counter(link.relation for i in train for link in rows[i].event_relations)
        heldrel=Counter(link.relation for i in held for link in rows[i].event_relations)
        statuses={e.factuality for i in train for e in rows[i].events}
        rej=sum(a.polarity=="REJECTED" for i in train for a in rows[i].arguments)
        if (not all(trrel[k]>0 for k in rel)
                or not {"ASSERTED","NEGATED","CONDITIONAL","FUTURE_UNREALIZED"}<=statuses
                or rej==0):
            raise ValueError("S22 missing semantic class in TRAIN side of held source fold")
        if axis=="CONSTRUCTION" and group<24:
            assert sum(heldrel.values())==8
    return rel,rejection


def manifest():
    rows=build_s22_train256()
    relation,rejection=verify_coverage(rows)
    digest=canonical_train_only_sha(rows)
    if FROZEN_S22_TRAIN256_SHA256 and digest!=FROZEN_S22_TRAIN256_SHA256:
        raise ValueError("S22 immutable TRAIN256 graph SHA drift")
    factuality=Counter(e.factuality for row in rows for e in row.events)
    return {
        "stage":STAGE,
        "source_sha256":digest,
        "sha_pinned_and_required_before_new_CPU_capture": bool(FROZEN_S22_TRAIN256_SHA256),
        "rows_synthetic_TRAIN_ONLY":EXPECTED,
        "lexical_family_count":FAMILY_COUNT,
        "surface_construction_count":PATTERN_COUNT,
        "grouped_TRAIN_only_folds":len(splits()),
        "positive_event_relations":dict(relation),
        "role_rejection_gold_links":rejection,
        "factuality_gold_events":dict(factuality),
        "event_count":sum(len(row.events) for row in rows),
        "directed_role_count":sum(len(row.arguments) for row in rows),
        "coreference_count":sum(len(row.coreferences) for row in rows),
        "minimal_if_vs_when_pairs":len(COUNTERFACTUAL_PAIRS)*FAMILY_COUNT,
        "each_relation_in_at_least_six_distinct_constructions":True,
        "each_rejected_role_in_four_distinct_constructions":True,
        "new_true43m_GT_CPU_not_yet_captured":True,
        "original_616_J7_not_replayed_in_this_stage":True,
        "no_model_forward_no_training_no_independent_EVAL":True,
        "no_claim_of_natural_language_generalization":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "LIVE_Render_Supabase_Jessica_APK_untouched":True,
    }


def text_only_inputs():
    return tuple(row.text for row in build_s22_train256())
