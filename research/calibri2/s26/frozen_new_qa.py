"""S26 sealed independent QA: 96 NEVER-SEEN lexical color-noun combinations.

Precommitted before **any** S26 training code, checkpoint, calibration,
selection, or results. Four wholly new lexical nouns, each with two
colors held out from TRAIN for authentic compositional transfer.
The 96 questions are disjoint from S20 and S24 QA, and S26 TRAIN may
NOT read or fit to them. S24 serves as baseline; after S26, score
the ONE precommitted checkpoint on these 96 + 48 controls + 16 open.

Always score the entire generated two-word answer AND actual EOS.
First token, forced-prefix token or rank alone IS NOT a correct response.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from calibri2.s26.spec import COLORS, OLD_NOUNS, EXTRA_NOUNS, RESERVED

VERSIONS = 12

def main_rows():
    rows=[]
    for k,(color,noun) in enumerate(RESERVED):
        foil_color=COLORS[(COLORS.index(color)+1+(k%2))%len(COLORS)]
        foil_noun=OLD_NOUNS[(k*5+2)%len(OLD_NOUNS)]
        for v in range(VERSIONS):
            owner=f"SealedQWynn{k:02d}{v:02d}"
            other=f"SealedQKian{k:02d}{v:02d}"
            label=f"{color} {noun}"
            foil=f"{foil_color} {foil_noun}"
            style=v%6
            if style==0:
                question=(f"At a tool festival {owner} picked a {label}; "
                          f"{other} picked a {foil}. What did {owner} pick? "
                          "Reply color then object in exactly two words.")
            elif style==1:
                question=(f"Two photographs are described: the item near "
                          f"{owner} is a {label}, and the item near {other} "
                          f"is a {foil}. Identify {owner}'s item with two words.")
            elif style==2:
                question=(f"Catalogue entry {v}: {other} = {foil}; "
                          f"{owner} = {label}. Return only the two-word "
                          f"description attached to {owner}.")
            elif style==3:
                question=(f"An assistant lists a {foil} for {other} and "
                          f"a {label} for {owner}. State {owner}'s color "
                          "and noun. No explanation.")
            elif style==4:
                question=(f"One box is marked '{owner}: {label}'. Another "
                          f"box is marked '{other}: {foil}'. "
                          f"Write the color and object in {owner}'s box.")
            else:
                question=(f"In a classroom report, {owner} is assigned "
                          f"the {label}; {other} is assigned the {foil}. "
                          f"What was assigned to {owner}? Answer in two words.")
            rows.append({
                "id":f"S26_NEW_{k:02d}_{v:02d}",
                "domain":"reserved_novel_color_noun",
                "question":question, "gold":label,
                "heldout":True, "novel_noun":noun, "novel_color":color
            })
    assert len(rows)==96
    return rows

def controls():
    rows=[]
    for i in range(24):
        c=COLORS[(i*3+i//5)%4]
        foil=COLORS[(COLORS.index(c)+2)%4]
        owner=f"SealedCiana{i:02d}"
        rows.append({
           "id":f"S26_COLOR_{i:02d}",
           "domain":"color_one_word_control",
           "question":f"At a marketplace {owner} sees a {c} ball while "
                      f"someone else sees a {foil} ball. "
                      f"What color is {owner}'s ball? One word.",
           "gold":c, "heldout":True
        })
    for i in range(24):
        yes=i%2==0
        a=10007+i*127
        b=a-(1+i%9) if yes else a+(1+i%9)
        rows.append({
            "id":f"S26_NUM_{i:02d}",
            "domain":"number_one_word_control",
            "question":f"A sensor registers {a} for A and {b} for B. "
                       "Is A larger than B? Reply Yes or No.",
            "gold":"Yes" if yes else "No", "heldout":True
        })
    return rows

OPEN_QUESTIONS=(
    "A red fork is on the table. Name the item in two words.",
    "What two words describe a blue glove in a shop?",
    "There is a green shirt in a window. Give its color and object.",
    "Someone has a yellow key. Describe it using two words.",
    "One character has a red fork, another has a blue fork. Identify the first item.",
    "A blue glove belongs to Ali and a red glove belongs to Bea. Describe Ali's glove.",
    "Describe how someone might use a fork, in one simple sentence.",
    "What could happen if someone loses a key? One sentence.",
    "A shirt and a glove are in a wardrobe. Briefly describe them.",
    "Explain a useful purpose for a glove in one sentence.",
    "What is 28 plus 16? Give the answer.",
    "Is 42 larger than 47? Answer Yes or No.",
    "Write a short sentence involving a spoon and a fork.",
    "A yellow key is found next to a green key. Which is yellow?",
    "Tell one sentence about a child's favorite shirt.",
    "What can be done with a missing key? One sentence.",
)

def cases():
    rows=main_rows()+controls()
    assert len(rows)==144
    assert len(OPEN_QUESTIONS)==16
    assert len({r["id"] for r in rows})==len(rows)
    assert len({r["question"] for r in rows})==len(rows)
    assert len(RESERVED)==8
    assert all((r["novel_color"],r["novel_noun"]) in RESERVED
               for r in main_rows())
    counts=Counter((r["novel_color"],r["novel_noun"]) for r in main_rows())
    assert set(counts)==set(RESERVED)
    assert all(n==12 for n in counts.values())
    return rows

def manifest():
    rows=cases()
    material=json.dumps({
        "questions":rows,
        "open":OPEN_QUESTIONS,
        "reserved":RESERVED
    },sort_keys=True,ensure_ascii=False,separators=(",",":"))
    sha=hashlib.sha256(material.encode("utf-8")).hexdigest()
    return {
        "stage":"S26_PRECOMMITTED_96_NOVEL_COMBINATIONS_48_CONTROLS_16_OPEN",
        "qa_suite_sha256":sha,
        "new_lexical_nouns":list(EXTRA_NOUNS),
        "reserved_color_noun_combinations":list(RESERVED),
        "n_novel_qa":96,"n_controls":48,"n_open":16,
        "cases":rows,
        "open_questions":OPEN_QUESTIONS,
        "frozen_before_S26_training_or_checkpoint_selection":True,
        "do_not_train_on_this_file_or_use_for_hyperparameter_selection":True,
        "complete_generated_answer_and_EOS_required":True
    }

if __name__=="__main__":
    root=Path("results/s26/frozen")
    root.mkdir(parents=True,exist_ok=True)
    report=manifest()
    out=root/"S26_FROZEN_96_NOVEL_NOUN_QA_MANIFEST.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in
        ("cases","open_questions")},ensure_ascii=False))
