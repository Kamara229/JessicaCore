"""Read-only posthoc structural answer extraction.

Retrospective on v190 raw answers. A new score here CANNOT replace the
precommitted v190 score or be advertised as independent blind QA.
For any sentence not matching narrowly scoped full-utterance patterns,
return UNGRADED, not wrong. Intentional preference for false negatives
over incidental correct word false positives.
"""
import re

COLORS=frozenset(("purple","orange","green","blue","yellow","pink","brown",
 "black","gray","white","red","turquoise","silver","violet","gold","maroon"))
# Recognizable complete factual statements: whole content and a single
# captured asserted answer, never arbitrary gold-word presence.
FACT_TEMPLATES=(
 r"the planet closest to the sun is called (?P<value>[a-z]+)",
 r"the red planet is called (?P<value>[a-z]+)",
 r"a regular hexagon has (?P<value>\d+) sides",
 r"the chemical formula of water is (?P<value>[a-z0-9]+)",
 r"the largest surface area of the ocean is the ocean of the (?P<value>[a-z]+) ocean",
 r"red and blue paint are mixed to create (?P<value>[a-z]+)",
 r"the instrument measuring temperature is the (?P<value>[a-z]+)",
 r"the smallest prime number is (?P<value>\d+)",
 r"the sahara desert is located in (?P<value>[a-z]+)",
 r"the planet known for prominent rings is (?P<value>[a-z]+)",
 r"there are (?P<value>\d+) days in a gregorian leap year",
 r"the gas humans need for aerobic respiration is (?P<value>[a-z]+)",
 r"the chemical element represented by the symbol au is (?P<value>[a-z]+)",
)
def full_utterance(text):
    if not isinstance(text,str):
        raise TypeError("Expected original model string")
    stripped=text.strip().lower()
    if len(stripped)>350 or any(x in stripped for x in ("\n","\r")):
        return None
    return stripped.rstrip(".! ").strip()

def _arithmetic_answer(question,rendered):
    task=re.fullmatch(
        r"compute (\d+) (plus|minus) (\d+)\. reply with the result as one number\.",
        question.casefold())
    if not task:return None
    a,action,b=task.groups()
    symbol="+" if action=="plus" else "-"
    patterns=(
        rf"{a}\s*\{symbol}\s*{b}\s*=\s*(?P<value>[-+]?\d+)",
        rf"{a}\s+{action}\s+{b}\s+(?:equals|is)\s+(?P<value>[-+]?\d+)",
    )
    for expression in patterns:
        match=re.fullmatch(expression,rendered)
        if match:return (match.group("value"),"whole_original_operand_arithmetic_assertion")
    return None

def _binding_answer(question,rendered):
    task=re.match(r"in a fictional workshop, ([a-z0-9]+) owns the ",
                  question.casefold())
    if not task:return None
    owner=re.escape(task.group(1))
    patterns=(
        rf"{owner}'s folder is (?:a )?(?P<value>[a-z]+)(?: folder)?",
        rf"{owner} owns the (?P<value>[a-z]+) folder",
    )
    for expression in patterns:
        match=re.fullmatch(expression,rendered)
        if match and match.group("value") in COLORS:
            return (match.group("value"),"specific_subject_possessive_claim")
    return None

def _factual_answer(question,rendered):
    for template in FACT_TEMPLATES:
        match=re.fullmatch(template,rendered)
        if match:
            return (match.group("value"),"entire_single_factual_assertion")
    return None

def grade_structured(case,old):
    """Posthoc result may only supplement—not mutate—v190 old grade."""
    pre=old["strict_grade"]
    if pre["machine_status"]!="ungraded":
        return {"case_id":case["case_id"],"machine_status":pre["machine_status"],
            "source":"original_precommitted_v190_grade",
            "recognized_answer":pre["recognized_answer"],
            "changed_from_original_v190":False}
    raw=old["response"]["verbatim_text"]
    rendered=full_utterance(raw)
    parsed=None
    if rendered is not None:
        domain=case["domain"]
        if domain=="arithmetic":parsed=_arithmetic_answer(case["question"],rendered)
        elif domain=="fictional_binding":parsed=_binding_answer(case["question"],rendered)
        elif domain=="factual_short":parsed=_factual_answer(case["question"],rendered)
    if parsed is None:
        return {"case_id":case["case_id"],"machine_status":"ungraded",
            "source":"no_full_safe_structural_parse", "recognized_answer":None,
            "changed_from_original_v190":False}
    value,source=parsed
    status="machine_correct" if value.casefold()==str(case["gold"]).casefold() else "machine_wrong"
    return {"case_id":case["case_id"],"machine_status":status,
        "source":source,"recognized_answer":value,
        "changed_from_original_v190":True}
