"""S29: SEALED PRE-TRAINING independent transfer QA, never gradient data.

The S26 evaluation has already been read by researchers and MUST NOT be
used for new checkpoint selection. This new S29 suite is frozen BEFORE any
S30-or-later training code or learning-rate choices are written. The eight
reserved NEW noun/color combinations can only appear in S29 QA, never
training labels or foil pairs. All valid outputs require COMPLETE answer+EOS,
never just a correct first token, teacher-forced continuation or mention.

This module is read-only QA specification; it does not load a model.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from calibri2.s26.spec import COLORS,ALL_NOUNS as S26_NOUNS

NEW_NOUNS=("bowl","ring","drum","rope")
# Opposite two colors for each noun stay TRAIN-eligible.
RESERVED_BY_NOUN={
    "bowl":("blue","yellow"),
    "ring":("red","green"),
    "drum":("blue","green"),
    "rope":("red","yellow"),
}
RESERVED=tuple((c,n) for n in NEW_NOUNS for c in RESERVED_BY_NOUN[n])
TRAIN_ELIGIBLE=tuple((c,n) for n in NEW_NOUNS for c in COLORS
                     if (c,n) not in RESERVED)
assert len(RESERVED)==len(TRAIN_ELIGIBLE)==8
assert not set(RESERVED)&set(TRAIN_ELIGIBLE)
assert not set(NEW_NOUNS)&set(S26_NOUNS)
assert set(RESERVED_BY_NOUN)==set(NEW_NOUNS)

OLD_CONTROLS=("cup","book","bag","lamp","chair","hat",
              "plate","scarf","clock","spoon","pencil","bottle")
PER_COMBINATION=12

def novel_96():
    rows=[]
    for k,(color,noun) in enumerate(RESERVED):
        # All distractors are OLD nouns; every new withheld answer is
        # visibly quoted in QUESTION as a genuine compositional task.
        foil_color=COLORS[(COLORS.index(color)+2+(k%2))%4]
        foil_noun=OLD_CONTROLS[(k*7+3)%len(OLD_CONTROLS)]
        label=f"{color} {noun}"
        distractor=f"{foil_color} {foil_noun}"
        for v in range(PER_COMBINATION):
            person=f"NovelTara{k:02d}{v:02d}"
            foil_person=f"NovelMilo{k:02d}{v:02d}"
            style=v%6
            if style==0:
                q=(f"At the train station {person} packed the {label}; "
                   f"{foil_person} packed the {distractor}. "
                   f"Which item did {person} pack? Give exactly two words.")
            elif style==1:
                q=(f"Lost and found records {foil_person}: {distractor}. "
                   f"Another record reads {person}: {label}. "
                   f"Return just the color and item from {person}'s record.")
            elif style==2:
                q=(f"On a museum shelf the {label} sits beside {person}, "
                   f"but the {distractor} sits beside {foil_person}. "
                   f"Identify the item beside {person} in two words.")
            elif style==3:
                q=(f"Two delivery labels were scanned. "
                   f"Label for {person} shows {label}; label for "
                   f"{foil_person} shows {distractor}. "
                   f"Reply with the exact color and object for {person}.")
            elif style==4:
                q=(f"After a photo shoot {person} described the {label} "
                   f"while {foil_person} described the {distractor}. "
                   f"What did {person} describe? Two words only.")
            else:
                q=(f"An inventory map links {foil_person} to the "
                   f"{distractor}, then links {person} to the "
                   f"{label}. Which two-word object belongs to {person}?")
            rows.append({"id":f"S29_NOVEL_{k:02d}_{v:02d}",
                         "domain":"S29_reserved_new_noun_color",
                         "question":q,"gold":label,
                         "novel_noun":noun,"novel_color":color,
                         "heldout":True,"split":"QA"})
    return rows

def controls_48():
    rows=[]
    for i in range(24):
        c=COLORS[(i*5+i//7)%4]
        other=COLORS[(COLORS.index(c)+2)%4]
        rows.append({"id":f"S29_COLOR_{i:02d}",
            "domain":"S29_color_one_word",
            "question":(f"In a science room {i+73} the color on "
                        f"Verena{i:02d}'s notebook is {c}. "
                        f"A different notebook is {other}. "
                        f"What is the first notebook's color? One word only."),
            "gold":c,"heldout":True,"split":"QA"})
    for i in range(24):
        yes=i%2==0
        a=15017+131*i
        offset=2+i%11
        b=a-offset if yes else a+offset
        rows.append({"id":f"S29_NUM_{i:02d}",
            "domain":"S29_numeric_yes_no",
            "question":(f"System channel X reads {a}; channel Y reads {b}. "
                        "Is channel X greater than channel Y? "
                        "Reply with only Yes or No."),
            "gold":"Yes" if yes else "No",
            "heldout":True,"split":"QA"})
    return rows

OPEN_16=(
    "A green ring lies on the shelf. Name it in two words.",
    "A blue bowl was broken. What object and color are mentioned? Two words.",
    "A yellow drum is in a room. Describe it in two words.",
    "A red rope hangs by the window. What item is there? Two words.",
    "Explain in one short sentence what a rope can be used for.",
    "What purpose could a bowl serve? One simple sentence.",
    "A ring and a drum are lying on a table. Describe the scene in one sentence.",
    "A chef uses a bowl. Describe the chef's action briefly.",
    "Is 105 larger than 117? Reply Yes or No.",
    "Is 291 smaller than 324? Reply Yes or No.",
    "Calculate 37 plus 25. Give the number only.",
    "Calculate 81 minus 29. Give the number only.",
    "A yellow rope and a blue ring were found. Which object is blue?",
    "A green drum is near a red drum. Identify the first item in two words.",
    "How can someone find an object they lost? One sentence.",
    "Write one clear sentence about preparing a museum exhibit.",
)

def cases():
    rows=novel_96()+controls_48()
    if len(rows)!=144 or len(OPEN_16)!=16:
        raise AssertionError("S29 immutable 96+48+16 contract")
    if len({r["id"] for r in rows})!=144 or len({r["question"] for r in rows})!=144:
        raise AssertionError("Duplicate S29 QA")
    counts=Counter((r["novel_color"],r["novel_noun"]) for r in novel_96())
    if set(counts)!=set(RESERVED) or set(counts.values())!={12}:
        raise AssertionError("Reserved new noun QA imbalance")
    if set(RESERVED)&set(TRAIN_ELIGIBLE):
        raise AssertionError("S29 QA labels leaked into TRAIN-eligible")
    return rows

def manifest():
    rows=cases()
    sealed={"questions":rows,"open":list(OPEN_16),
            "reserved":[list(pair) for pair in RESERVED],
            "new_nouns":list(NEW_NOUNS),
            "train_eligible":[list(pair) for pair in TRAIN_ELIGIBLE]}
    raw=json.dumps(sealed,sort_keys=True,ensure_ascii=False,
                   separators=(",",":"))
    return {
        "stage":"CALIBRI_S29_FROZEN_BEFORE_NEXT_TRAINING",
        "frozen_sha256":hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "n_new_noun_binding":96,"n_control":48,"n_open":16,
        "new_nouns":list(NEW_NOUNS),
        "reserved_color_new_nouns":[list(pair) for pair in RESERVED],
        "train_eligible_color_new_nouns":[list(pair) for pair in TRAIN_ELIGIBLE],
        "all_QA":rows,"open_QA":list(OPEN_16),
        "S26_QA_already_inspected_and_not_reused_for_training":True,
        "S29_evaluation_new_and_frozen_before_any_subsequent_training":True,
        "no_checkpoints_or_hyperparameters_selected_using_these_answers":True,
        "require_entire_model_generated_answer_plus_EOS":True,
        "do_not_import_this_module_into_training":True,
    }

if __name__=="__main__":
    r=manifest()
    p=Path("results/s29/frozen")
    p.mkdir(parents=True,exist_ok=True)
    (p/"S29_PRECOMMITTED_INDEPENDENT_96_48_16.json").write_text(
        json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in r.items() if k not in
         ("all_QA","open_QA")},ensure_ascii=False),flush=True)
