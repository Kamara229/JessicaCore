"""S126 progressive-capacity PRECOMMITMENT (no model weights created).

The baseline is a *random-initialized*, compressed 36M SmolLM2-style
Llama with 49,152 tied embeddings; it is not a compressed copy of the
HuggingFace pretrained 135M model. Geometry changes must be trained
and independently proven: a larger tensor shape is NOT a better model.

Physical bpp must be computed from the ACTUAL complete serialized
physical payload, including separate GT and controller sidecars and
configuration, divided by the UNIQUE logical model parameters.
Speed must be benchmarked with actual CPU decoding, same hardware.
"""
from __future__ import annotations
from dataclasses import dataclass

VOCAB=49152
HEAD_DIM=64
REFERENCE_FP32_UNIQUE_PARAMS=134_515_008
MAX_BPP=0.700
MIN_CPU_RATIO=0.90
S101_ORIGINAL_456_FULL=353
S101_ORIGINAL_456_FIRST=387


@dataclass(frozen=True)
class Geometry:
    stage: str
    width: int
    ffn: int
    layers: int
    heads: int
    kv_heads: int

    def unique_parameters(self)->int:
        if (self.width<=0 or self.ffn<=0 or self.layers<=0 or self.heads<=0
                or self.kv_heads<=0 or self.heads%self.kv_heads
                or self.width//self.heads!=HEAD_DIM
                or self.width%self.heads):
            raise ValueError("S126 invalid tied-embedding Llama geometry")
        h=self.width
        # Tied embedding/head, GQA K,V = kv_heads * 64.
        # Per decoder: attention Q,O,K,V, SwiGLU (gate/up/down), 2 RMS norms.
        return (VOCAB*h+self.layers*(
            2*h*h+2*h*self.kv_heads*HEAD_DIM+3*h*self.ffn+2*h)+h)

    def max_complete_archive_bytes(self)->int:
        return int(self.unique_parameters()*MAX_BPP//8)


ROADMAP=(
    Geometry("S126-A_current_36m",320,1536,12,5,1),
    Geometry("S126-B_40m",320,1536,14,5,1),
    Geometry("S126-C_43m",320,1536,16,5,1),
    Geometry("S126-D_53m",384,1536,16,6,2),
    Geometry("S126-E_63m",384,1792,18,6,2),
    Geometry("S126-F_74m",384,2048,20,6,2),
    Geometry("S126-G_86m",448,2048,20,7,1),
    Geometry("S126-H_99m",448,2048,24,7,1),
    Geometry("S126-I_116m",512,2048,24,8,2),
    Geometry("S126-J_135m",512,2048,29,8,2),
)


def validate_precommitted_geometry():
    numbers=[x.unique_parameters() for x in ROADMAP]
    if numbers[0]!=36_380_480 or not all(a<b for a,b in zip(numbers,numbers[1:])):
        raise AssertionError("S126 size schedule or original S101 shape drifted")
    if abs(numbers[-1]-REFERENCE_FP32_UNIQUE_PARAMS)/REFERENCE_FP32_UNIQUE_PARAMS>0.01:
        raise AssertionError("Final CALIBRI does not match 135M FP32 size")
    # Roadmap only: not approved, no claims about actual compression,
    # throughput or retention until measured for each true checkpoint.
    return [{"stage":x.stage,"unique_params":x.unique_parameters(),
             "maximum_ALL_physical_bytes":x.max_complete_archive_bytes(),
             "theoretical_only":True} for x in ROADMAP]


@dataclass(frozen=True)
class PhysicalAdmission:
    checked_old_train:int
    old_full_lost:int
    old_first_lost:int
    complete_physical_bytes:int
    unique_model_parameters:int
    cpu_ratio_vs_frozen_S101:float
    immutable_source_verified:bool
    # New TRAIN gains guide research but cannot replace independent QA.
    new_train_full_gain:int

    def eligible_for_RESEARCH_checkpoint(self)->bool:
        import math
        return (self.checked_old_train==456
                and self.old_full_lost==0 and self.old_first_lost==0
                and self.complete_physical_bytes>0
                and self.unique_model_parameters>0
                and self.immutable_source_verified
                and math.isfinite(self.cpu_ratio_vs_frozen_S101)
                and self.cpu_ratio_vs_frozen_S101>=MIN_CPU_RATIO
                and 8*self.complete_physical_bytes<=
                    MAX_BPP*self.unique_model_parameters
                and self.new_train_full_gain>=0)


validate_precommitted_geometry()
