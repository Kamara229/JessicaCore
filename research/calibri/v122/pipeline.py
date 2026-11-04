"""Real 135M compressed-native controller trained in 8MB, frozen for future hosts."""
from __future__ import annotations
import argparse,hashlib,json,copy
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v118.archive import Paid
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from .protocol import VALID,TRAIN,OPEN,MAX_CHECKPOINT_BYTES,V119_SOURCE_RUN
from .policy import initialize,SharedGreenPolicy,serialize
from .fit import fit
from .measure import evaluate,speed

def main(host_path,critic_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    paid=Paid(host_path.read_bytes())
    if (hashlib.sha256(paid.base).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("pretrained original 135M source unexpectedly changed")
    base=out/"base_immutable.cb59"
    base.write_bytes(paid.base)
    model,physical,_=make_model("baseline",SimpleNamespace(original=base,folder=out))
    if not physical["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("original pretrained model had hidden 2D dense matrices")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    policy=initialize(critic_path)
    initial=copy.deepcopy(policy).eval()
    with torch.no_grad():
        for p in initial.parameters():p.requires_grad_(False)
    before=evaluate(model,tok,paid,initial)
    history=fit(model,tok,paid,policy,out)
    dest=out/"GreenTunnel_shared_10MB_6gates.safetensors"
    sidecar=serialize(policy,dest)
    restored=SharedGreenPolicy()
    restored.load_state_dict(load_file(str(dest)),strict=True)
    restored.eval()
    if any(torch.isnan(p).any() or torch.isinf(p).any() for p in restored.parameters()):
        raise AssertionError("nonfinite neural controller weights")
    after=evaluate(model,tok,paid,restored)
    shuffled=evaluate(model,tok,paid,restored,conditions="shuffled")
    # Genuine complete autonomous answer parity across un/serialized model.
    in_memory=evaluate(model,tok,paid,policy)
    if [x["generated_ids"] for x in after["qa"]+after["open"]] != [
        x["generated_ids"] for x in in_memory["qa"]+in_memory["open"]]:
        raise AssertionError("physically saved controller lost output behavior")
    original_speed=speed(model,tok,paid,initial)
    learned_speed=speed(model,tok,paid,restored)
    report={
        "experiment":"v122 model-size-agnostic 8MB Green Tunnel policy on actual 135M frozen native source",
        "whole_source_CB59_SHA256_frozen":ORIGINAL_CB59_SHA,
        "unmodified_original_135m_host_full_archive_bytes":len(paid.raw),
        "declared_base_model_bpp_EXCLUDES_ONE_SHARED_GREEN_MODULE":paid.stats()["all_in_bpp"],
        "user_reporting_policy":"Core bpp counts core model only. Reusable Green Tunnel is reported separately by real bytes.",
        "shared_neural_controller":sidecar,
        "standalone_weight_limit_bytes":MAX_CHECKPOINT_BYTES,
        "physical_runtime_total_core_plus_shared_controller_bytes":len(paid.raw)+sidecar["green_tunnel_full_neural_weight_bytes"],
        "physical_runtime_combined_bpp_IF_counted_together":8*(len(paid.raw)+sidecar["green_tunnel_full_neural_weight_bytes"])/TOTAL_UNIQUE,
        "initial_zero_gate_policy":before,
        "training":history,
        "genuine_autoregressive_learned_policy":after,
        "prompt_control_shuffled_ablation":shuffled,
        "paired_real_CPU":{
            "initial_policy":original_speed,"trained_policy":learned_speed,
            "trained_over_initial_CPU_ratio":learned_speed["tokens_per_second_including_controller"]/
                 original_speed["tokens_per_second_including_controller"]},
        "frozen_neural_policy_safetensors_saved":True,
        "no_model_specific_trainable_input_dimension_in_controller":True,
        "real_transfer_to_another_pretrained_LLM_NOT_TESTED":True,
        "trillion_parameter_transfer_NOT_PROVEN":True,
        "v89_official_independent_TEST_unopened":True,
        "quality_sci_PASS":False,
        "production_main_Render_Supabase_Jessica_APK_UNTOUCHED":True}
    (out/"v122_shared_10mb_policy.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":"v122_real_shared_neural_controller_completed",
        "host_core_bpp_excluding_shared_tunnel":paid.stats()["all_in_bpp"],
        "shared_tunnel_bytes":sidecar["green_tunnel_full_neural_weight_bytes"],
        "base_correct":before["summary"]["strict_correct_and_eos"],
        "trained_correct":after["summary"]["strict_correct_and_eos"],
        "shuffled_correct":shuffled["summary"]["strict_correct_and_eos"],
        "trained_vs_untrained_CPU_ratio":report["paired_real_CPU"]["trained_over_initial_CPU_ratio"],
        "learned_prompt_mean_gate_spread":after["summary"]["different_prompt_mean_gate_spread"],
        "no_quality_claim":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("host",type=Path)
    p.add_argument("old_critic",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();main(a.host,a.old_critic,a.out)
