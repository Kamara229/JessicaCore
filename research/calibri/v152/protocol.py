"""Fixed sensitivity study before looking at any CB151 QA or data.

Oracle FP32 layer swaps are deliberately NOT compact. This test identifies
which original 30 transformer blocks' quantization drives teacher KL on
eight previously used diagnostic prompts, not independent TEST or VALID.
"""
from v132.protocol import all_examples
from v117.protocol import ORIGINAL_CB59_SHA
from v141.run import ORIGINAL_V140_SHA

LAYERS=30
DIAGNOSTIC_QUESTION_INDICES=(0,2,4,6,8,10,12,14)
ARMS=("FP32_teacher","genuine_packed_CB59")+tuple(f"oracle_FP32_layer_{i:02d}" for i in range(LAYERS))
SOURCE_CB147_SHA="639be185c259260c614823831275113ffa2d83231eb54805ebfc67be30da5bbd"
NO_QA_CLAIMS=True


def locked_questions():
    samples=tuple(all_examples())
    if len(samples)!=16:
        raise ValueError("unexpected predeclared diagnostic corpus")
    return tuple(samples[i] for i in DIAGNOSTIC_QUESTION_INDICES)


def check():
    if LAYERS!=30 or len(ARMS)!=32 or len(set(ARMS))!=32:
        raise ValueError("not one complete 30-layer oracle scan")
    if DIAGNOSTIC_QUESTION_INDICES!=(0,2,4,6,8,10,12,14):
        raise ValueError("single-block selection changed after observing teacher")
    if any(i<0 or i>=16 for i in DIAGNOSTIC_QUESTION_INDICES):
        raise ValueError("diagnostic indices out of bounds")
    return True
