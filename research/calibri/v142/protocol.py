"""Fixed two domain VALID windows; no tuning or source-weight edits."""
from v117.protocol import ORIGINAL_CB59_SHA
from v141.run import ORIGINAL_V140_SHA

DOMAINS=("wiki_valid","news_valid")
WINDOW=64
WINDOWS_PER_DOMAIN=2
ARMS=("FP32","CB59","CB140_v141_LUT")
MODELS_BYTES=("original_CB59","full_CB140_paid")
VALID_SEED=142
assert len(DOMAINS)==2
assert WINDOWS_PER_DOMAIN==2


def pairs(token_ids):
    if len(token_ids)<WINDOW*WINDOWS_PER_DOMAIN:
        raise ValueError("missing frozen validation tokens")
    return tuple(token_ids[i*WINDOW:(i+1)*WINDOW].unsqueeze(0)
                 for i in range(WINDOWS_PER_DOMAIN))
