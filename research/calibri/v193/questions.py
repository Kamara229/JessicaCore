"""Completely NEW, precommitted 96 public questions.

Original FP32 baselines must be loaded only AFTER questions and the
v192 exact grader SHA manifest are sealed to files. This is a public
development sample and never the user-private v89 independent TEST.
"""
import hashlib
from collections import Counter,defaultdict

from .protocol import COUNT,DOMAIN_COUNTS

FACTUAL=(
 ("Which planet is the largest in the Solar System? Reply with one word.","Jupiter"),
 ("How many legs does a typical adult spider have? Reply with a number.","8"),
 ("What is the chemical symbol for silver? Reply with one symbol.","Ag"),
 ("Which element has the chemical symbol K? Reply with one word.","potassium"),
 ("What is the capital city of Norway? Reply with one city.","Oslo"),
 ("What is the capital city of Portugal? Reply with one city.","Lisbon"),
 ("What is the capital city of Greece? Reply with one city.","Athens"),
 ("Which body organ pumps blood through arteries? Reply with one word.","heart"),
 ("Which process turns sunlight into chemical energy in plants? Reply with one word.","photosynthesis"),
 ("What is the SI unit named for force? Reply with one word.","newton"),
 ("Which animal is known for having a long trunk? Reply with one word.","elephant"),
 ("Which instrument is commonly used to measure atmospheric pressure? Reply with one word.","barometer"),
 ("What is the chemical symbol for copper? Reply with one symbol.","Cu"),
 ("How many degrees are in a full rotation? Reply with a number.","360"),
 ("Which geometric solid has six square faces of equal size? Reply with one word.","cube"),
 ("Which mammal is fastest on land? Reply with one word.","cheetah"),
 ("Which ocean lies between Africa and Australia? Reply with one word.","Indian"),
 ("Which color results from mixing yellow and blue paint? Reply with one word.","green"),
 ("Which star is at the center of our Solar System? Reply with one word.","Sun"),
 ("What is the chemical symbol of neon? Reply with one symbol.","Ne"),
 ("How many chambers are in a typical human heart? Reply with a number.","4"),
 ("How many continents are counted in the seven-continent model? Reply with a number.","7"),
 ("What is the smallest unit of a chemical element? Reply with one word.","atom"),
 ("Which gas makes up the largest fraction of Earth's atmosphere? Reply with one word.","nitrogen"),
)
assert len(FACTUAL)==24

def cases():
    items=[]
    def push(domain,question,gold,pair=None):
        assert question and gold is not None
        items.append({"case_id":len(items),"domain":domain,"question":question,
            "gold":str(gold),"pair_id":pair,
            "question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest()})
    for i in range(8):
        a=329+i*27;b=48+i*7
        push("arithmetic",f"Compute {a} plus {b}. Reply with the result as one number.",a+b)
    for i in range(8):
        a=920+i*37;b=217+i*11
        push("arithmetic",f"Compute {a} minus {b}. Reply with the result as one number.",a-b)
    for i in range(8):
        a=12+i*2;b=17+(i%5)
        push("arithmetic",f"Compute {a} times {b}. Reply with the result as one number.",a*b)
    for i in range(12):
        larger=1103+i*29;smaller=larger-(13+i)
        pair=f"greater_{i:02d}"
        push("opposite_yesno",
             f"Is {larger} greater than {smaller}? Reply with only Yes or No.","Yes",pair)
        push("opposite_yesno",
             f"Is {smaller} greater than {larger}? Reply with only Yes or No.","No",pair)
    first=("purple","green","blue","yellow","pink","brown","black",
           "gray","red","turquoise","silver","gold")
    second=("white","orange","maroon","violet","cyan","beige","teal",
            "navy","magenta","blue","green","black")
    for i in range(12):
        owner=f"CoraMelu{i}";other=f"TaviNero{i}"
        pair=f"binding_{i:02d}"
        a,b=first[i],second[i]
        for color_one,color_other in ((a,b),(b,a)):
            push("fictional_binding",
                f"In a fictional workshop, {owner} owns the {color_one} folder and {other} owns the {color_other} folder. What color is {owner}'s folder? Reply with one color.",
                color_one,pair)
    for question,gold in FACTUAL:push("factual_short",question,gold)
    if len(items)!=COUNT or dict(Counter(x["domain"] for x in items))!=DOMAIN_COUNTS:
        raise AssertionError("Wrong complete frozen original QA domain counts")
    ids=[x["question_sha256"] for x in items]
    if len(set(ids))!=COUNT:raise AssertionError("Repeated v193 prompts")
    for domain in ("opposite_yesno","fictional_binding"):
        paired=defaultdict(list)
        for x in items:
            if x["domain"]==domain:paired[x["pair_id"]].append(x["gold"])
        if len(paired)!=12 or any(len(v)!=2 or v[0]==v[1] for v in paired.values()):
            raise AssertionError("Missing or unbalanced opposing pair")
    return items
