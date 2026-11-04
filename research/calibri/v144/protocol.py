"""Prior-selection and strictly later confirmation source windows."""
from v142.protocol import DOMAINS,WINDOW

GAIN_VALUES=(0.0,.25,.5,.75,1.0)
GAIN_GRID=tuple((i,o) for i in GAIN_VALUES for o in GAIN_VALUES)
SELECT_WINDOW_IDS=(0,1)
CONFIRM_WINDOW_IDS=(2,3)
assert len(GAIN_GRID)==25
assert not set(SELECT_WINDOW_IDS)&set(CONFIRM_WINDOW_IDS)


def fixed_input_windows(token_ids):
    if len(token_ids)<(max(CONFIRM_WINDOW_IDS)+1)*WINDOW:
        raise ValueError("TRAIN-split source VALID has fewer than 4 windows")
    return tuple(token_ids[k*WINDOW:(k+1)*WINDOW][None,:]
                 for k in (*SELECT_WINDOW_IDS,*CONFIRM_WINDOW_IDS))


def label(input_gain,head_gain):
    if (input_gain,head_gain) not in GAIN_GRID:
        raise ValueError("gain arm not fixed before holdout")
    return "input_%d_head_%d"%(round(input_gain*100),round(head_gain*100))


BASELINE=label(0.0,0.0)
ORIGINAL_PAID=label(1.0,1.0)
