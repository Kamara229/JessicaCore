"""CALIBRI v169: 96 fully generated QA prompts, two presentation formats.

A physically packed compact source must answer questions, not merely put
the same mass on logits as the teacher. This benchmark deliberately does
NOT score KL/NLL/PPL and never fits weights to question labels.
"""
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
ARMS=("FP32","CB59","CB161")
DOMAINS=("arithmetic","binding","yesno","factual","logic")
DOMAIN_COUNTS={"arithmetic":16,"binding":32,"yesno":16,"factual":16,"logic":16}
FORMATS=("direct","brief")
MAX_NEW_TOKENS=32
FULL_ARCHIVE_BYTES=11292174
V161_SUCCESS_RUN=41920207168
EXPECTED_BASE_TASKS=96
EXPECTED_GENERATIONS_PER_ARM=192

def check():
    if sum(DOMAIN_COUNTS.values())!=EXPECTED_BASE_TASKS:
        raise ValueError("changed number of actual text questions")
    if len(FORMATS)!=2 or len(ARMS)!=3:
        raise ValueError("missing model/formats")
    if EXPECTED_GENERATIONS_PER_ARM!=EXPECTED_BASE_TASKS*len(FORMATS):
        raise ValueError("missing generated texts")
    if 8*FULL_ARCHIVE_BYTES/TOTAL_UNIQUE>=0.7:
        raise ValueError("physical source violates compressed objective")
    if MAX_NEW_TOKENS!=32:
        raise ValueError("changed generation cap")
    return True
