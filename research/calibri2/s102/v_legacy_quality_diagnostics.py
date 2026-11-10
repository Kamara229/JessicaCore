"""S102 V-series informed frozen-TRAIN full-answer diagnostics.

Adapted *principles*, not weights/QA, from V128 lexical token bug,
V165 top-K ceiling, V222/V225 first divergence/late loop causal path,
V226/V228 gold-valid repetition and V231 choice collapse. This is a pure
read-only audit. No private test, no model selection or inference lookup.
"""
from __future__ import annotations

from collections import Counter
import math


def _first_divergence(gold, got):
    for i in range(max(len(gold), len(got))):
        if i >= len(gold) or i >= len(got) or gold[i] != got[i]:
            return i
    return None


def _unwanted_adjacent_repeat(gold, got):
    """A repeat is not harmful merely because neighboring tokens match.

    Count only adjacent repeat at a position where the gold target is not
    itself the identical repeated token at that position.
    """
    return sum(
        got[i] == got[i-1] and
        (i >= len(gold) or i-1 >= len(gold) or
         gold[i] != gold[i-1] or got[i] != gold[i])
        for i in range(1, len(got))
    )


def sequence_diagnostics(expected, old, new, routed, *, tok=None, copy_notes=None):
    keys=set(expected)
    if len(keys)!=456 or keys!=set(old) or keys!=set(new) or keys!=set(routed):
        raise AssertionError("Complete 456 frozen TRAIN identities required")
    position=Counter()
    losses_by_profile=Counter()
    gains_by_profile=Counter()
    first_output_old=Counter()
    first_output_new=Counter()
    bad_repeat_old=0
    bad_repeat_new=0
    wrong_early_eos_old=0
    wrong_early_eos_new=0
    lexical_empty_expected=0
    lexical_empty_old=0
    lexical_empty_new=0
    lost_ids=[]
    gained_ids=[]
    case={}
    copy_stat=Counter()
    for ident in sorted(keys):
        gold=list(expected[ident])
        before=list(old[ident])
        after=list(new[ident])
        profile=routed[ident]
        if not gold or gold[-1] is None:
            raise AssertionError("Gold token target including EOS missing")
        if profile not in {"S87","S91","S93"}:
            raise AssertionError("Unknown frozen S95 profile")
        first_output_old[str(before[0]) if before else "empty"]+=1
        first_output_new[str(after[0]) if after else "empty"]+=1
        if tok is not None:
            if not tok.decode(gold[:1],skip_special_tokens=True).strip():
                lexical_empty_expected+=1
            if not tok.decode(before[:1],skip_special_tokens=True).strip():
                lexical_empty_old+=1
            if not tok.decode(after[:1],skip_special_tokens=True).strip():
                lexical_empty_new+=1
        bad_repeat_old+=_unwanted_adjacent_repeat(gold,before)
        bad_repeat_new+=_unwanted_adjacent_repeat(gold,after)
        eos=gold[-1]
        wrong_early_eos_old+=int(eos in before[:len(gold)-1])
        wrong_early_eos_new+=int(eos in after[:len(gold)-1])
        was=before==gold
        now=after==gold
        if was and not now:
            idx=_first_divergence(gold,after)
            position["first" if idx==0 else "second" if idx==1
                     else "EOS" if idx==len(gold)-1 else "late"]+=1
            losses_by_profile[profile]+=1
            lost_ids.append(ident)
        elif now and not was:
            gained_ids.append(ident)
            gains_by_profile[profile]+=1
        if copy_notes is not None:
            note=copy_notes.get(ident)
            if note is None:
                raise AssertionError("Copy diagnostics absent for TRAIN identity")
            for variant in ("old","new"):
                stat=note[variant]
                if stat.get("neural_override_applied"):
                    copy_stat[f"{variant}_overrides"]+=1
                if stat.get("candidate_present"):
                    copy_stat[f"{variant}_eligible"]+=1
        pair=ident.rsplit("/",1)[0]
        side=ident.rsplit("/",1)[1]
        if side not in ("left","right"):
            raise AssertionError("Expected one TRAIN pair side")
        case.setdefault(pair,{})[side]=(was,now)
    both_old=both_new=0
    for pair,rows in case.items():
        if set(rows)!={"left","right"}:
            raise AssertionError("TRAIN paired contrast is incomplete")
        both_old+=int(rows["left"][0] and rows["right"][0])
        both_new+=int(rows["left"][1] and rows["right"][1])
    # These are diagnostics, NOT semantic correctness; no QA/test is read.
    return {
        "event":"S102_V_LEGACY_FULL_SEQUENCE_DIAGNOSTICS",
        "checked_TRAIN":len(keys),
        "protected_old_full_EOS":sum(old[k]==expected[k] for k in keys),
        "new_full_EOS":sum(new[k]==expected[k] for k in keys),
        "lost_full_count":len(lost_ids),
        "gained_full_count":len(gained_ids),
        "lost_full_TRAIN_identities":lost_ids,
        "gained_full_TRAIN_identities":gained_ids,
        "lost_first_divergence_position":dict(position),
        "lost_full_by_frozen_route":dict(losses_by_profile),
        "gained_full_by_frozen_route":dict(gains_by_profile),
        "pairs_old_both_correct":both_old,
        "pairs_new_both_correct":both_new,
        "spurious_adjacent_repeats_old":bad_repeat_old,
        "spurious_adjacent_repeats_new":bad_repeat_new,
        "wrong_early_EOS_old":wrong_early_eos_old,
        "wrong_early_EOS_new":wrong_early_eos_new,
        "lexically_empty_first_gold":lexical_empty_expected,
        "lexically_empty_first_old":lexical_empty_old,
        "lexically_empty_first_new":lexical_empty_new,
        "most_common_first_old":first_output_old.most_common(5),
        "most_common_first_new":first_output_new.most_common(5),
        "copy_policy":dict(copy_stat),
        "no_question_text_or_answers_exported":True,
        "never_use_for_QA_checkpoint_selection":True
    }


def synthetic_regressions():
    expected={str(i)+"/"+side:[10,11,2] for i in range(228)
              for side in ("left","right")}
    old={k:list(v) for k,v in expected.items()}
    new={k:list(v) for k,v in old.items()}
    route={k:"S93" for k in expected}
    # Actual gold is identical at no adjacent position; new spurious
    # repeat must be correctly diagnosed at the first differing position.
    new["0/left"]=[10,10,2]
    new["1/left"]=[10,11,11,2]
    got=sequence_diagnostics(expected,old,new,route)
    assert got["lost_full_count"]==2
    assert got["lost_first_divergence_position"]=={"second":1,"EOS":1}
    assert got["spurious_adjacent_repeats_new"]==2
    assert got["spurious_adjacent_repeats_old"]==0
    assert got["pairs_old_both_correct"]==228
    assert got["pairs_new_both_correct"]==226
    return {"event":"S102_V_LEGACY_GOLD_AWARE_SEQUENCE_DIAGNOSTIC_UNIT_PASS",
            "gold_aware_repeat":True,"late_first_divergence":True,
            "paired_both_correct":True}
