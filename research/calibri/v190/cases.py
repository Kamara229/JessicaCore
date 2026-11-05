"""New frozen public prompts, unrelated to v169 historical 96 development prompts.

The original two checkpoints are only EVALUATED, never fitted. Every
question's gold and id is fixed before either model is loaded.
"""
import hashlib
import random
from v169.sources import all_cases
from . import protocol as p

FACTS=(
 ("Which planet is closest to the Sun? Answer with only its name.","Mercury"),
 ("Which chemical element is represented by the symbol Fe? Answer with one word.","iron"),
 ("Which planet is famously called the Red Planet? Answer with one word.","Mars"),
 ("How many sides does a regular hexagon have? Answer with one number.","6"),
 ("What is the chemical formula of water? Answer with only the formula.","H2O"),
 ("Which ocean has the largest surface area? Answer with one word.","Pacific"),
 ("How many degrees is one right angle? Answer with a number.","90"),
 ("What do artists get by mixing red and blue paint? Answer with one color.","purple"),
 ("Which egg-laying Australian mammal has a duck-like bill? Answer with one word.","platypus"),
 ("Which instrument measures temperature? Answer with one word.","thermometer"),
 ("What is the smallest prime number? Answer with one number.","2"),
 ("On which continent is the Sahara Desert located? Answer with one word.","Africa"),
 ("Which planet is known for prominent rings? Answer with one word.","Saturn"),
 ("How many days are in a Gregorian leap year? Answer with one number.","366"),
 ("Which gas do humans need for aerobic respiration? Answer with one word.","oxygen"),
 ("Which chemical element is represented by the symbol Au? Answer with one word.","gold"),
)


def cases():
    rng=random.Random(p.SEED)
    items=[]
    def append(domain,question,answer,pair=None):
        items.append({"case_id":len(items),"domain":domain,"question":question,
            "gold":str(answer),"pair_id":pair,
            "question_sha256":hashlib.sha256(question.encode()).hexdigest()})
    for i in range(8):
        a=200+i*17
        b=37+i*3
        append("arithmetic",f"Compute {a} plus {b}. Reply with the result as one number.",a+b)
    for i in range(8):
        a=760+i*19
        b=213+i*7
        append("arithmetic",f"Compute {a} minus {b}. Reply with the result as one number.",a-b)
    for i in range(8):
        larger=517+i*23
        smaller=larger-(6+i)
        pair=f"numeric_{i:02d}"
        append("opposite_yesno",
            f"Is {larger} greater than {smaller}? Reply with only Yes or No.","Yes",pair)
        append("opposite_yesno",
            f"Is {smaller} greater than {larger}? Reply with only Yes or No.","No",pair)
    colors=("purple","orange","green","blue","yellow","pink","brown","black")
    others=("gray","white","red","turquoise","silver","violet","gold","maroon")
    for i in range(8):
        owner=f"NeriDalo{i}";other=f"JumaTeso{i}"
        pair=f"folder_{i:02d}"
        a,b=colors[i],others[i]
        append("fictional_binding",
            f"In a fictional workshop, {owner} owns the {a} folder and {other} owns the {b} folder. What color is {owner}'s folder? Reply with one color.",a,pair)
        append("fictional_binding",
            f"In a fictional workshop, {owner} owns the {b} folder and {other} owns the {a} folder. What color is {owner}'s folder? Reply with one color.",b,pair)
    for question,gold in FACTS:
        append("factual_short",question,gold)
    expected={name:p.DOMAIN_SIZE for name in p.DOMAINS}
    actual={name:sum(x["domain"]==name for x in items) for name in p.DOMAINS}
    if actual!=expected or len(items)!=p.TOTAL:raise AssertionError("Unbalanced real QA source")
    old_questions={x["question"] for x in all_cases()}
    new_questions={x["question"] for x in items}
    if len(new_questions)!=len(items) or new_questions & old_questions:
        raise AssertionError("Question reuse from old historically tuned public DEV")
    for domain in ("opposite_yesno","fictional_binding"):
        pairs={}
        for x in items:
            if x["domain"]==domain:
                pairs.setdefault(x["pair_id"],[]).append(x)
        if len(pairs)!=8 or any(len(rows)!=2 or rows[0]["gold"]==rows[1]["gold"]
             for rows in pairs.values()):
            raise AssertionError("Each frozen opposing pair must have different correct answers")
    return items
