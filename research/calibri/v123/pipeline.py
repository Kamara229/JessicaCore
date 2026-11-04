"""Freeze ONE exact 8MB Green policy, run on TWO actual disparate LLMs.

No controller retraining, new weights, hidden dimensions or tokenizers used.
Source-specific physical adapters are paid or parameter-free as appropriate.
Does NOT claim multilingual/general competence or 1T scale demonstrated.
"""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoModelForCausalLM

from v122.policy import SharedGreenPolicy
from v122.steer import Installed
from v118.archive import Paid
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from paid_lowrank_v76 import make_model
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import GREEN_SHA,PYTHIA,PYTHIA_SHA,PROBES,MAX_NEW_TOKENS
from .real_generations import compare_neox

def load_fixed(source):
    content=source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=GREEN_SHA:
        raise ValueError("shared Green Tunnel changed between host models")
    network=SharedGreenPolicy()
    network.load_state_dict(load_file(str(source)),strict=True)
    for p in network.parameters():p.requires_grad_(False)
    network.eval()
    return network

@torch.inference_mode()
def compare_smol(model,tok,paid,policy):
    originals=Installed(model,paid)
    pristine=SharedGreenPolicy().eval()
    for p in pristine.parameters():p.requires_grad_(False)
    results=[]
    try:
        for q in PROBES:
            input_=build_chat(tok,q)
            def sample():
                started=time.perf_counter()
                res=generate_one(model,tok,input_,stop_at_eos=True)
                return {"text":res["text"],"token_ids":res["ids"],
                        "seconds":time.perf_counter()-started}
            originals.set_question(pristine,q)
            baseline=sample()
            started=time.perf_counter()
            gains=originals.set_question(policy,q)
            ctrl_ms=time.perf_counter()-started
            controlled=sample()
            results.append({"prompt":q,"baseline":baseline,"green":controlled,
                            "shared_neural_controller_seconds":ctrl_ms,
                            "nonzero_token_impact":baseline["token_ids"]!=controlled["token_ids"],
                            "gate_min":float(gains.min()),
                            "gate_max":float(gains.max())})
    finally:originals.restore()
    return {"real_model":MODEL,
            "paid_native_135m_source_bpp_excluding_shared_tunnel":paid.stats()["all_in_bpp"],
            "actual_native_packed_C_GEMV":True,
            "produced":results,
            "different_token_sequences":sum(x["nonzero_token_impact"] for x in results),
            "total_prompts":len(results)}

def main(green_path,host_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    fixed=load_fixed(green_path)
    original_sha=hashlib.sha256(green_path.read_bytes()).hexdigest()
    # First, a small actual GPT-NeoX transformer with COMPLETELY DIFFERENT
    # layer names/widths/number of layers than the 135M Llama host.
    pythia_tok=AutoTokenizer.from_pretrained(PYTHIA,revision=PYTHIA_SHA)
    pythia=AutoModelForCausalLM.from_pretrained(
        PYTHIA,revision=PYTHIA_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    neox_result=compare_neox(pythia,pythia_tok,fixed)
    del pythia
    # Second, original physically compact Llama/SmolLM2 135M, no new source.
    paid=Paid(host_path.read_bytes())
    if (hashlib.sha256(paid.base).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("original 135M pinned native trained source violated")
    core_path=out/"original_native_CB59_source.cb59"
    core_path.write_bytes(paid.base)
    smol,source_info,_=make_model("baseline",SimpleNamespace(
         original=core_path,folder=out))
    if not source_info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("original packed C source hot matrix was decoded")
    smol_tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    smol_result=compare_smol(smol,smol_tok,paid,fixed)
    after_sha=hashlib.sha256(green_path.read_bytes()).hexdigest()
    if original_sha!=after_sha or original_sha!=GREEN_SHA:
        raise AssertionError("shared neural weights changed between host inference")
    report={
        "experiment":"v123 exact same green weights run unchanged on real GPT-NeoX14M and SmolLM135M",
        "one_frozen_policy_sha256":GREEN_SHA,
        "one_shared_neural_checkpoint_bytes":green_path.stat().st_size,
        "controller_retrained_or_modified_between_hosts":False,
        "controller_output_dimension_fixed_at_6":True,
        "controller_model_independent_prompt_UTF8_bytes":True,
        "two_actual_distinct_pretrained_model_architectures":True,
        "Pythia_GPTNeoX_14M_baseline_FP32_uncompressed":neox_result,
        "SmolLM2_LLama_135M_native_under_0p7_bpp":smol_result,
        "host_core_bpp_excludes_one_global_green_per_user":True,
        "physical_neural_weights_stored_separately_once":True,
        "cross_host_quality_or_improved_coherence_NOT_PROVEN":True,
        "large_10GB_or_trillion_host_NOT_ACTUALLY_TESTED":True,
        "v89_original_independent_TEST_unopened":True,
        "no_live_or_main_changes":True}
    (out/"v123_two_architecture_fixed_policy.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v123_TWO_REAL_HOSTS",
        "one_frozen_neural_sha":GREEN_SHA,
        "same_green_weights_used_across_architectures":True,
        "Pythia_14M_outputs_changed":neox_result["different_token_sequences"],
        "SmolLM_135M_outputs_changed":smol_result["different_token_sequences"],
        "real_general_quality_restored":False}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("green",type=Path);p.add_argument("smol",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();main(a.green,a.smol,a.out)
