"""Frozen controls on previously used diagnostic prompts only.

The original CB153 layer6 was selected by v152 FP32 layer oracle, but v153
new WikiText+AG News VALID had clear no-pass NLL/KL. Now isolate 7 original
paid 1-bit operator contributions on the SAME pre-existing v152 prompts.
No model fitting or new validation data are touched. Selection on these
diagnostic prompts would require truly new VALID confirmation afterward.
"""
from v152.protocol import locked_questions
from v117.protocol import ORIGINAL_CB59_SHA
LAYER=6
CB153_SHA="fda0f1c146d8d302186dbd7851caaf146c4babc71a13c1d8a49022da3397bba5"
MATRIX_KEYS=tuple(f"model.layers.{LAYER}.{name}.weight" for name in (
    "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
    "self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"))
ARMS=("CB59_original","CB153_all7")+tuple(
    f"only_{k.split('.')[-2]}" for k in MATRIX_KEYS)+tuple(
    f"except_{k.split('.')[-2]}" for k in MATRIX_KEYS)
EXPECTED_ARMS=16
EXPECTED_DIAGNOSTIC_PROMPTS=8
MIN_ORIGINAL_MATRIX=7


def check_protocol(paid):
    if paid.sha256!=CB153_SHA or paid.layer!=LAYER:
        raise ValueError("did not load pinned physically paid complete layer6 CB153")
    if set(paid.patches)!=set(MATRIX_KEYS) or len(MATRIX_KEYS)!=7:
        raise ValueError("not exactly the original seven linear operators")
    if len(ARMS)!=EXPECTED_ARMS or len(set(ARMS))!=EXPECTED_ARMS:
        raise ValueError("incomplete fixed operator subset")
    questions=locked_questions()
    if len(questions)!=EXPECTED_DIAGNOSTIC_PROMPTS:
        raise ValueError("v152 repeated diagnostic source changed")
    return questions
