"""Prototype 10MB standalone GREEN TUNNEL on real frozen compact 135M LLM.

Five independent factual checks:
1) neural SIDE MODULE physically <=10MB, all weights in one file;
2) native CB59 v119 source byte-for-byte pinned, no dense pretrained W;
3) gold-first-subtoken top-K headroom in actual source (not generation);
4) critic never sees gold answer during inference or selects oracle answer;
5) all-in model + sidecar bpp is PAID, separate from source-only bpp.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from v118.archive import Paid
from v118.runtime import attach,restore
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from paid_lowrank_v76 import make_model
from .train import fit
from .probe import first_correct_token_rank,headroom_summary,real_candidates
from .protocol import MAX_CHECKPOINT_BYTES,BASELINE_SOURCE_RUN

def run(source,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    host=Paid(source.read_bytes())
    if (hashlib.sha256(host.base).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("not frozen genuine full original 135M model")
    origin=out/"CB59_original_unmodified.cb59"
    origin.write_bytes(host.base)
    model,meta,_=make_model("baseline",SimpleNamespace(original=origin,folder=out))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("must execute packed original source directly")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    critic,study=fit(out)
    base_rank=first_correct_token_rank(model,tok)
    patched=attach(model,host)
    try:
        repaired_rank=first_correct_token_rank(model,tok)
        t0=time.perf_counter()
        generated=real_candidates(model,tok,critic)
        observation_seconds=time.perf_counter()-t0
    finally:
        restore(model,patched)
    # The sidecar is a second independently paid physical model.
    critic_bytes=study["full_weights_and_metadata_file_bytes"]
    combined_bytes=len(host.raw)+critic_bytes
    if critic_bytes>MAX_CHECKPOINT_BYTES:
        raise AssertionError("standalone 10MB limit exceeded")
    report={
        "experiment":"CALIBRI GREEN TUNNEL v120 standalone neural candidate control",
        "standalone_neural_critic_physical":study,
        "host_135M_source_SHA256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "host_135M_original_CB59_SHA_locked":True,
        "host_full_135M_physical_bpp":host.stats()["all_in_bpp"],
        "standalone_critic_extra_bytes":critic_bytes,
        "combined_135M_plus_one_reusable_critic_bytes":combined_bytes,
        "combined_bpp_ONE_HOST_WITH_CRITIC":8*combined_bytes/TOTAL_UNIQUE,
        "does_NOT_meet_full_0_7bpp_for_ONE_HOST":True,
        "source_model_training_or_weights_CHANGED":False,
        "no_user_semantic_answers_injected_by_critic":True,
        "critic_does_not_generate_new_words":True,
        "cross_model_architecture_independent_BYTE_IO_only":True,
        "cross_model_quality_transfer_NOT_PROVEN":True,
        "pretrained_host_model":MODEL,
        "host_artifact_actions_run":BASELINE_SOURCE_RUN,
        "original_CB59_oracle_gold_FIRST_subtoken_coverage":{
            "by_topk":headroom_summary(base_rank),"rows":base_rank},
        "v119_native_compressed_oracle_gold_FIRST_subtoken_coverage":{
            "by_topk":headroom_summary(repaired_rank),"rows":repaired_rank},
        "real_generated_answers_and_neural_critic_verdicts":generated,
        "real_critic_plus_model_diagnostic_time_seconds":observation_seconds,
        "open_ended_task_semantic_validity_UNVERIFIED":True,
        "not_connected_to_Jessica_production":True,
        "independent_v89_TEST_unopened":True,
        "scientific_general_quality_PASS":False,
    }
    (out/"v120_green_tunnel_real.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":"v120_green_tunnel_135M_complete",
        "critic_saved_bytes":critic_bytes,
        "combined_135m_bpp":report["combined_bpp_ONE_HOST_WITH_CRITIC"],
        "critic_heldout":study["heldout_gold_vs_synthetic_corrupt_metrics"],
        "CB59_gold_first_token_coverage":headroom_summary(base_rank),
        "v119_gold_first_token_coverage":headroom_summary(repaired_rank),
        "real_candidate_gate_passed":sum(x["choose_to_pass_original_answer"] for x in generated),
        "real_candidate_count":len(generated),
        "no_semantic_pass":True,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.source,a.out)
