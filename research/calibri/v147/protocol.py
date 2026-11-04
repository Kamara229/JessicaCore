"""Before fit or VALID, freeze original source and new disjoint checks."""
from v145.protocol import WIKI_REVISION,NEWS_REVISION
from v117.protocol import TOTAL_UNIQUE

FIT_DOMAINS=("wiki_fit","news_fit")
VALID_DOMAINS=("wiki_prior_valid","news_prior_valid")
WIKI_VALID_START=26624
WIKI_VALID_COUNT=512
NEWS_VALID_START=9500
NEWS_VALID_END=9600
VALID_CONTEXT=64
ORIGINAL_PARAMETERS=TOTAL_UNIQUE
assert WIKI_VALID_START>19968
assert NEWS_VALID_START>=9200
assert WIKI_VALID_COUNT//VALID_CONTEXT==8
