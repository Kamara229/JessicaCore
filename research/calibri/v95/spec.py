"""Frozen candidate-search split contract: all original public TRAIN, no official TEST."""
from __future__ import annotations

TARGET="model.layers.13.self_attn.v_proj"
TRAIN_BOOLQ=(1350,1382)
TRAIN_SCIQ=(1100,1124)
VALID_BOOLQ=(8350,8414)
VALID_SCIQ=(9100,9132)
TRAIN_GENERATION=(
    ("reasoning","There are nine pencils. Four are used. How many pencils remain? Answer:"),
    ("factual","Question: What is the largest planet in our solar system? Answer:"),
)
VALID_GENERATION=(
    ("reasoning","If two books cost three dollars each, the total cost is:"),
    ("factual","The gas plants absorb from the air during photosynthesis is:"),
    ("instruction","Write one helpful step to make a simple paper airplane:"),
    ("creative","The curious kitten discovered a mysterious red door and"),
    ("arithmetic","What is six multiplied by seven? Answer:"),
    ("reasoning","A shelf has five cups and three are taken away. Cups left:"),
    ("factual","The freezing point of fresh water in Celsius is:"),
    ("instruction","Give a practical tip for keeping a bicycle safe:"),
)
SPLITS={"BoolQ":{"TRAIN":TRAIN_BOOLQ,"TRAIN_VALID":VALID_BOOLQ},
        "SciQ":{"TRAIN":TRAIN_SCIQ,"TRAIN_VALID":VALID_SCIQ}}
OTHER_PAST={"BoolQ":((800,832),(6000,6032),(7200,7328)),
            "SciQ":((5200,5264),)}
def verify():
    for name,span in SPLITS.items():
        train,valid=span["TRAIN"],span["TRAIN_VALID"]
        if not (0<=train[0]<train[1] and 0<=valid[0]<valid[1]):
            raise AssertionError("no empty/negative public original TRAIN spans")
        for prior in OTHER_PAST[name]+(train,):
            if prior==train:continue
            if not (valid[1]<=prior[0] or valid[0]>=prior[1]):
                raise AssertionError("VALID overlaps TRAIN or older evaluated selection")
        for prior in OTHER_PAST[name]:
            if not (train[1]<=prior[0] or train[0]>=prior[1]):
                raise AssertionError("TRAIN overlaps prior selected records")
    assert len(TRAIN_GENERATION)==2 and len(VALID_GENERATION)>=8
    assert set(x[1] for x in TRAIN_GENERATION).isdisjoint(x[1] for x in VALID_GENERATION)
    return True
