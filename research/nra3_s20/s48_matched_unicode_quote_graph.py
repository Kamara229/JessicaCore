"""S48 structurally matched RAW Unicode quote interval graph (not a language oracle).

Pure raw-string, token-offset quote stack. Unlike S39's naive _inside_quotes
depth, this detector understands complementary open/close glyphs, nested
mixed pairs, German „…“, curly ‘…’, literal apostrophes inside words and
backslash-escaped literal quotation marks. It fails closed for any
unmatched outer opener; previously completed spans remain intact.
Its structural output is a raw text feature for a small TRAIN-only quote
readout, NOT human gold or active Calibri43M/NRA2 inference.

Designed before first S48 held26 evaluation and AFTER discovering S47
failure categories. Therefore improvements on S48 held are development
diagnostics, not a new unbiased general Russian NLP benchmark.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar

OPEN_TO_CLOSE = {"«":"»","“":"”","„":"“","‘":"’",'"':'"',"'":"'"}
CLOSERS = frozenset(OPEN_TO_CLOSE.values())
ALL_QUOTE_MARKS = frozenset(set(OPEN_TO_CLOSE) | set(CLOSERS))

@dataclass(frozen=True)
class QuoteStructure:
    content:torch.Tensor
    valid_outer_pairs:tuple[tuple[int,int],...]
    dangling_openers:int
    ignored_apostrophes:int
    ignored_escaped_quote_marks:int
    mismatched_closer_count:int


def _is_escaped(text, at):
    back=0
    at-=1
    while at>=0 and text[at]=="\\":
        back+=1
        at-=1
    return back%2==1


def _apostrophe_in_word(text,token):
    if token.form!="'":return False
    start,end=token.start,token.end
    return bool(start>0 and end<len(text)
                and text[start-1].isalpha() and text[end].isalpha())


def matched_quote_graph_from_raw(text):
    """No TRAIN annotations, rows, class labels or gold indexes accepted.

    Spans are counted as quoted only when the OUTERMOST parent quote
    closes with its exact expected closer. A nested mixed quote must
    close before the parent; any malformed enclosing span is discarded.
    Quotes are marked as delimiters, not quoted content. Non-quote
    punctuation inside a fully closed quote retains quoted membership.
    """
    tokens=raw_surface_grammar(text)
    n=len(tokens)
    if n<3 or n>40:
        raise ValueError("S48 shadow feature supports 3..40 Unicode tokens")
    stack=[]
    completed=[]
    delimiters=set()
    ignored_apostrophes=0
    ignored_escaped=0
    mismatched=0
    for i,t in enumerate(tokens):
        g=t.form
        if g not in ALL_QUOTE_MARKS:continue
        if _is_escaped(text,t.start):
            ignored_escaped+=1
            continue
        if _apostrophe_in_word(text,t):
            ignored_apostrophes+=1
            continue
        delimiters.add(i)
        if stack and stack[-1][1]==g:
            # Close innermost quote, preserving the enclosing block.
            opener,_=stack.pop()
            if not stack:
                completed.append((opener,i))
        elif g in OPEN_TO_CLOSE:
            # Ambiguous German closing “ is handled by matching top first.
            stack.append((i,OPEN_TO_CLOSE[g]))
        else:
            # Stray closer or mismatched nested closer: do not auto-close.
            mismatched+=1
    membership=torch.zeros(n,dtype=torch.float32)
    for lo,hi in completed:
        for i in range(lo+1,hi):
            if i not in delimiters:
                membership[i]=1.
    return QuoteStructure(
        content=membership,
        valid_outer_pairs=tuple(completed),
        dangling_openers=len(stack),
        ignored_apostrophes=ignored_apostrophes,
        ignored_escaped_quote_marks=ignored_escaped,
        mismatched_closer_count=mismatched)
