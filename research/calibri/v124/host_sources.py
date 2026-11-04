"""Actual independent LLM candidates; positive gold NEVER supplied to generator."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v118.archive import Paid
from v118.runtime import attach,restore
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import PYTHIA,PYTHIA_SHA,MAX_NEW_TOKENS_PYTHIA

@torch.inference_mode()
def generate_hard_negatives(tasks,hostfile,out):
    queries=[q for q,a in tasks]
    # The 14M Pythia source uses plain causal text, never given references.
    tok=AutoTokenizer.from_pretrained(PYTHIA,revision=PYTHIA_SHA)
    pythia=AutoModelForCausalLM.from_pretrained(
        PYTHIA,revision=PYTHIA_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    neox={}
    for q in queries:
        ids=tok(q,return_tensors="pt")["input_ids"]
        seq=pythia.generate(ids,max_new_tokens=MAX_NEW_TOKENS_PYTHIA,
               do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        neox[q]=tok.decode(seq[0,ids.shape[1]:],skip_special_tokens=True)
    del pythia
    # Genuine native-packed source + frozen paid v119 correction factors.
    paid=Paid(hostfile.read_bytes())
    if (hashlib.sha256(paid.base).hexdigest()!=ORIGINAL_CB59_SHA
         or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("original real pretrained packed source modified")
    frozen=out/"immutable_packed_CB59_for_hard_negative_generation.cb59"
    frozen.write_bytes(paid.base)
    model,physical,_=make_model("baseline",SimpleNamespace(original=frozen,folder=out))
    if not physical["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("real host source not directly compact")
    source_tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    handles=attach(model,paid)
    try:
        smol={}
        for q in queries:
            response=generate_one(model,source_tok,build_chat(source_tok,q),stop_at_eos=True)
            smol[q]=response["text"]
    finally:
        restore(model,handles)
    return {"Pythia14M":neox,"CB59_v119":smol},{
        "pretrained_models":["EleutherAI/pythia-14m GPTNeoX FP32",
                             "HuggingFaceTB/SmolLM2-135M-Instruct native CB59"],
        "training_or_reference_answers_seen_by_generator":False,
        "source_CB59_original_SHA_locked":True,
        "number_of_questions_per_architecture":len(queries),
        "host_unique_reference_text_values_not_persisted_in_input":True,
    }
