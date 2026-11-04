"""Lock the 30 original layer activation comparison before opening TRAIN FIT.

The v157 32-window fresh source check FAILED: v_proj correction worsened
WikiText true NLL +0.0372 and teacher KL +0.0404, and both domains failed
the 20/32-window + leave-best-out robustness gate. v158 found upstream
input drift ~2.7-2.9x weight-only v_proj error on existing TRAIN FIT.
Consequently test upstream drift beginning at the original tied embedding,
through all 30 original transformer layers, before testing another bit
allocation. This is NOT official independent v89 TEST nor unseen VALID.

Only original v76 *already-used* wiki_fit and news_fit source windows are
allowed; no fitting, tuning, bitstream recompression, QA labels or TEST.
"""
from v154.protocol import CB153_SHA
from v117.protocol import ORIGINAL_CB59_SHA
from v147.protocol import WIKI_REVISION,NEWS_REVISION
from v158.protocol import FIT_DOMAINS,WINDOW,ORIGINAL_TRAIN_FIT_WINDOWS_EACH

LAYERS=30
TOKENS_EACH=ORIGINAL_TRAIN_FIT_WINDOWS_EACH*WINDOW
MODEL_ARMS=("FP32_teacher","original_native_CB59","paid_layer6_vproj")
EXPERIMENTAL_PAID_PATH="model.layers.6.self_attn.v_proj.weight"
TRACE_KEYS=("input_embedding",)+tuple(f"block_{i:02d}_output" for i in range(LAYERS))+("final_norm",)
EXPECTED_STAGES=LAYERS+2
BOUNDED_TRACES_ONLY=True


def check_protocol():
    if (LAYERS,TOKENS_EACH,EXPECTED_STAGES)!=(30,256,32):
        raise ValueError("source geometry changed")
    if FIT_DOMAINS!=("wiki_fit","news_fit") or WINDOW!=64:
        raise ValueError("TRAIN FIT provenance changed")
    if len(set(TRACE_KEYS))!=EXPECTED_STAGES or (
            TRACE_KEYS[0]!="input_embedding" or
            TRACE_KEYS[7]!="block_06_output" or
            TRACE_KEYS[-1]!="final_norm"):
        raise ValueError("missing source 30-layer trace points")
    if MODEL_ARMS!=("FP32_teacher","original_native_CB59","paid_layer6_vproj"):
        raise ValueError("original model comparator changed")
    return True
