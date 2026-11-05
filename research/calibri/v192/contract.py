"""Versioned QA grading contract; freeze before any v193 original model generation.

Conservative, syntax-based, FULL-UTTERANCE (not semantic LLM judge).
No model weights, no original answers, no private test referenced.
"""
VERSION="v192-locked-grader-v1"
DOMAINS=("arithmetic","opposite_yesno","fictional_binding","factual_short")
STATES=("machine_correct","machine_wrong","ungraded")
MAX_ANSWER_CHARS=400
SYMBOLS={"+":"plus","-":"minus","*":"times"}
COLORS=frozenset(("purple","orange","green","blue","yellow","pink",
   "brown","black","gray","grey","white","red","turquoise","silver",
   "violet","gold","maroon","beige","cyan","magenta","teal","navy"))
REFUSALS=frozenset(("unknown","unsure","not sure","i don't know",
   "i cannot tell","i can't tell","maybe","perhaps","none","idk","sorry"))
BPP_CEILING_NUMERATOR=7
BPP_CEILING_DENOMINATOR=80
assert len(set(DOMAINS))==4 and len(STATES)==3
