"""Frozen whole-utterance syntactic candidate extraction, no correct-answer lookup.

Only recognize a single reply whose complete text asserts exactly one
answer. Recognizing a candidate does not imply correctness: comparison
to the pre-frozen gold happens in the next facade module.

No model inference and no version-specific answer artifact reads.
"""
import re

from . import contract as C

# Allowed finite, *non-semantic* answer lead-ins. The remainder must
# satisfy the same strict full-utterance recognizer as a bare answer.
PREFIXES=(
    r"the\s+answer\s+is\s+",
    r"the\s+answer\s*:\s*",
    r"answer\s*:\s*",
    r"the\s+result\s+is\s+",
    r"result\s*:\s*",
    r"it\s+is\s+",
    r"it's\s+",
)
# Older v191 used specific full factual claims; here frozen *before
# future v193 answers* with conservative reusable whole-claim forms.
FACT_ASSERTIONS=(
    r"(?:the )?planet closest to (?:the )?sun is (?:called )?(?P<value>[a-z]+)",
    r"(?:the )?red planet is (?:called )?(?P<value>[a-z]+)",
    r"(?:a )?regular hexagon has (?P<value>\d+) sides",
    r"(?:the )?chemical formula (?:for|of) water is (?P<value>[a-z0-9]+)",
    r"(?:the )?largest ocean is (?:the )?(?P<value>[a-z]+)(?: ocean)?",
    r"(?:red and blue paint )?mix(?:ing|ed) (?:red and blue paint )?(?:make|makes|gives|produce|produces|creates?) (?P<value>[a-z]+)",
    r"(?:the )?instrument (?:that )?measures temperature is (?:the )?(?P<value>[a-z]+)",
    r"(?:the )?smallest prime number is (?P<value>\d+)",
    r"(?:the )?sahara desert is (?:located )?in (?P<value>[a-z]+)",
    r"(?:the )?planet (?:known )?for (?:its )?(?:prominent )?rings is (?P<value>[a-z]+)",
    r"(?:there are )?(?P<value>\d+) days (?:are )?in (?:a )?(?:gregorian )?leap year",
    r"(?:the )?gas (?:that )?humans need for (?:aerobic )?respiration is (?P<value>[a-z]+)",
    r"(?:the )?chemical element (?:represented by )?(?:the )?symbol (?:fe|au) is (?P<value>[a-z]+)",
    r"(?:the )?capital of [a-z]+ is (?P<value>[a-z]+)",
    r"(?:the )?(?:right|perpendicular) angle is (?P<value>\d+) degrees",
)

def normalize(raw):
    """Neutral text cleanup; never strip clauses or remove negations."""
    if not isinstance(raw,str):
        raise TypeError("Expected generated answer text")
    if not raw or len(raw)>C.MAX_ANSWER_CHARS or any(x in raw for x in ("\n","\r")):
        return None
    text=re.sub(r"\s+"," ",raw.strip().casefold())
    if not text:return None
    text=text.rstrip(".! ").strip()
    for part in PREFIXES:
        candidate=re.sub(r"^"+part,"",text,count=1)
        if candidate!=text:
            text=candidate
            break
    text=text.strip().strip(chr(34)+chr(39)+" ").rstrip(".! ").strip(chr(34)+chr(39)+" ")
    return text or None

def _single(domain,text):
    """Bare single answer; exactly one numeric/YesNo/color/word claim."""
    if domain=="arithmetic" and re.fullmatch(r"[-+]?\d{1,12}",text):
        return (str(int(text)),"bare_numeric_answer")
    if domain=="opposite_yesno" and text in ("yes","no"):
        return (text,"bare_boolean_answer")
    if domain=="fictional_binding" and text in C.COLORS:
        return (text,"bare_color_answer")
    if domain=="factual_short" and text not in C.REFUSALS and (
          re.fullmatch(r"\d{1,8}",text) or
          re.fullmatch(r"[a-z][a-z0-9-]{0,31}",text)):
        return (text,"bare_single_factual_token")
    return None

def _arithmetic(question,text):
    # Phrase-bound reference operands: cannot return another problem's
    # correct result or a superficially plausible but different equation.
    q=re.fullmatch(
      r"(?:compute|calculate) (\d+) (plus|minus|times) (\d+)\. "
      r"reply with (?:the result as one number|only the answer|one integer)\.",
      question.casefold())
    if q is None:
        q=re.fullmatch(r"what is (\d+) (plus|minus|times) (\d+)\? reply with one integer\.",
                       question.casefold())
    if q is None:return None
    a,operation,b=q.groups()
    op={"plus":r"\+","minus":"-","times":r"\*" }[operation]
    patterns=(
      rf"{a}\s*{op}\s*{b}\s*=\s*(?P<value>[-+]?\d+)",
      rf"{a}\s+{operation}\s+{b}\s+(?:equals|is)\s+(?P<value>[-+]?\d+)",
    )
    for p in patterns:
        m=re.fullmatch(p,text)
        if m:return (str(int(m.group("value"))),"complete_question_matched_equation")
    return None

def _binding(question,text):
    q=re.match(r"in a fictional workshop, ([a-z0-9]+) owns the ",question.casefold())
    if not q:return None
    owner=re.escape(q.group(1))
    patterns=(
      rf"{owner}'s folder is (?:a )?(?P<value>[a-z]+)(?: folder)?",
      rf"{owner} owns the (?P<value>[a-z]+) folder",
    )
    for p in patterns:
        m=re.fullmatch(p,text)
        if m and m.group("value") in C.COLORS:
            return (m.group("value"),"complete_asked_owner_binding")
    return None

def _factual(text):
    for p in FACT_ASSERTIONS:
        m=re.fullmatch(p,text)
        if m:return (m.group("value"),"complete_bounded_factual_assertion")
    return None

def recognize(domain,question,raw):
    if domain not in C.DOMAINS:raise ValueError("Unknown frozen grading domain")
    text=normalize(raw)
    if text is None or text in C.REFUSALS:return None
    parsed=_single(domain,text)
    if parsed is not None:return parsed
    if domain=="arithmetic":return _arithmetic(question,text)
    if domain=="fictional_binding":return _binding(question,text)
    if domain=="factual_short":return _factual(text)
    return None
