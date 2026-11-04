"""Pre-registered BEFORE independent TRAIN-split source evaluation.

v144 chose input gain =0, output gain =0.5 from its source TRAIN VALID
windows 0,1 and passed untouched windows 2,3. These new data ranges have
NOT been used by v144 selection or its followup confirmation.
"""
from v144.protocol import label

SOURCE_DOMAINS=("wiki_shift","news_shift")
SELECTED=(0.0,0.5)
BASELINE=(0.0,0.0)
PAID=(1.0,1.0)
ARMS=(label(*BASELINE),label(*SELECTED),label(*PAID))
WINDOW=64
WINDOWS=8
TOKEN_COUNT=WINDOW*WINDOWS

WIKI_REVISION="b08601e04326c79dfdd32d625aee71d232d685c3"
NEWS_REVISION="eb185aade064a813bc0b7f42de02595523103ca4"
WIKI_SOURCE_START=19456
NEWS_SOURCE_START=9100
NEWS_SOURCE_END=9200

# Avoid the exact original v76 and v144 WIN spans and record ranges.
OLD_WIKI_FIT=(12288,12544)
OLD_WIKI_VALID=(14336,14592)
OLD_NEWS_FIT=(8000,8100)
OLD_NEWS_VALID=(8500,8600)


def disjoint(span,other):
    return span[1]<=other[0] or other[1]<=span[0]


def verify_predeclared():
    if ARMS!=(label(0.,0.),label(0.,.5),label(1.,1.)):
        raise ValueError("v144 selected arm altered before new data")
    wiki=(WIKI_SOURCE_START,WIKI_SOURCE_START+TOKEN_COUNT)
    news=(NEWS_SOURCE_START,NEWS_SOURCE_END)
    if not all(disjoint(wiki,earlier) for earlier in (OLD_WIKI_FIT,OLD_WIKI_VALID)):
        raise ValueError("overlapping old WikiText examples")
    if not all(disjoint(news,earlier) for earlier in (OLD_NEWS_FIT,OLD_NEWS_VALID)):
        raise ValueError("overlapping old news records")
    if (WIKI_REVISION,NEWS_REVISION)!=(
            "b08601e04326c79dfdd32d625aee71d232d685c3",
            "eb185aade064a813bc0b7f42de02595523103ca4"):
        raise ValueError("data snapshot changed")
    return True


def windows(tokens):
    if len(tokens)!=TOKEN_COUNT:
        raise ValueError("new dataset must provide exactly 512 tokens")
    return tuple(tokens[i*WINDOW:(i+1)*WINDOW][None,:] for i in range(WINDOWS))
