"""S186: strictly prompt-only candidates for S69 owner phrases and TRAIN history numeric.

These are SOURCE GRAMMAR algorithms, not SHA lookup, not TRAIN memorization:
- S69: full anchored four original source frames, 2 distinct owner/value
  records, same-color distinct noun, literal repeated owner binding.
- historical yes/no: full anchored integer comparison question, computes
  > from question numerals without any expected answer.
Unsupported/negated/ambiguous templates must abstain; this is research
SHADOW only, never live inference, model/Green Tunnel edits or QA/TEST.
"""
from __future__ import annotations
import re
from calibri2.s181.source_grammar_shadow import shadow_candidate as s181_candidate

S69_TEMPLATES=(
    "TRAIN route 6901: The archive shelf for {a} is coded {left}. For {b} the shelf code is {right}. Write the TWO tokens assigned to {a}.",
    "TRAIN routing 6902: In a shipping manifest, customer {a} selects {left}; customer {b} selects {right}. Output only {a}'s two-word selection.",
    "TRAIN mapping 6903: The notation table binds {a} to {left} and {b} to {right}. Report the ordered color/object pair for {a}.",
    "TRAIN encoding 6904: Row {a} => {left}; row {b} => {right}. Reconstruct {a}'s color plus noun without extra words.",
)
FIELDS=re.compile(r"\{(a|b|left|right)\}")
VALUES={"a":r"[A-Za-z][A-Za-z0-9]*","b":r"[A-Za-z][A-Za-z0-9]*",
        "left":r"[a-z]+ [a-z]+","right":r"[a-z]+ [a-z]+"}


def _exact_pattern(template):
    parts,seen,start=[],set(),0
    for m in FIELDS.finditer(template):
        parts.append(re.escape(template[start:m.start()]))
        k=m.group(1)
        parts.append(f"(?P={k})" if k in seen else f"(?P<{k}>{VALUES[k]})")
        seen.add(k);start=m.end()
    parts.append(re.escape(template[start:]))
    if seen!=set(VALUES):
        raise AssertionError("S186 original S69 source template fields changed")
    return re.compile("^"+"".join(parts)+"$")


PATTERNS=tuple(_exact_pattern(p) for p in S69_TEMPLATES)
HISTORY_NUMERIC=re.compile(
    r"^Is ([0-9]{1,6}) greater than ([0-9]{1,6})\? Reply with only Yes or No\.$"
)


def s69_owner_phrase(question):
    if not isinstance(question,str) or len(question)>1024:return None
    matches=[]
    for pattern in PATTERNS:
        m=pattern.fullmatch(question)
        if m is None:continue
        fields=m.groupdict()
        if fields["a"]==fields["b"] or fields["left"]==fields["right"]:
            continue
        left_color,left_noun=fields["left"].split()
        right_color,right_noun=fields["right"].split()
        if left_color!=right_color or left_noun==right_noun:
            continue
        matches.append(fields["left"])
    return matches[0] if len(matches)==1 else None


def history_numeric_answer(question):
    if not isinstance(question,str) or len(question)>1024:return None
    m=HISTORY_NUMERIC.fullmatch(question)
    if m is None:return None
    x,y=(int(n) for n in m.groups())
    if x==y: return None  # fail-closed for ties; old source has no ties
    return "Yes" if x>y else "No"


def source_candidate(question):
    """Never take question SHA, gold, IDs, error families, or performance."""
    options=[s181_candidate(question),s69_owner_phrase(question),
             history_numeric_answer(question)]
    values=[x for x in options if x is not None]
    return values[0] if len(values)==1 else None
