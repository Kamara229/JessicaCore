"""NRA2 S3E: capture full 43M frozen CPU layer12 token states <-> graphs.

Repeat exactly 616 S194 original CPU J7 generated tokens. Inside the same
immutable S190 S93 CPU phase, add 48 genuinely *new* synthetic graph prompt
PREFILLS ONLY (no generated candidate answer, no target or graph label passed).
The last decoder layer read-only hook captures contextual 320D activations.
Only AFTER all CPU work, materialize 32 TRAIN / 16 structurally novel graph
labels. Freeze Calibri, Green Tunnel and independent 2,241-param graph NRA.
Gate=0. Research only, no QA/TEST, LIVE, Render, Supabase or APK.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file,load_file

from calibri2.s194.actual_cpu_j7_bridge import real_j7_baseline,output_digest
from nra2_bridge.stage3e_token_features import FrozenAllTokenFeatures
from nra2_bridge.passive_final_layer import NRA2PassiveFinalLayer
from nra2_bridge.stage3c_graph_data import (
    build_graph_rows,build_graph_targets,no_answer_template_lookup,
    TRAIN_COUNT,HELD_COUNT,TOTAL_COUNT,
)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if len(sys.argv)<15:
        raise SystemExit("S3E expects entire original 616 real S194 native CPU J7 CLI")
    argv=list(sys.argv)
    out=Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3E cannot overwrite research evidence")

    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    s3c=json.loads(Path("inputs/nra2_stage3c/NRA2_S3C_GRAPH48_REAL_CPU_PAIRS_AUDIT.json").read_text())
    s3a=json.loads(Path("inputs/nra2_stage3a/NRA2_S3A_REAL_CPU_32_TRAIN_FEATURE_AUDIT.json").read_text())
    s3b=json.loads(Path("inputs/nra2_stage3b/NRA2_S3B_24TRAIN_8HELDOUT_PROXY_AUDIT.json").read_text())
    standalone=json.loads(Path("inputs/nra2/NRA2_STANDALONE_REPORT.json").read_text())
    learned=Path("inputs/nra2/NRA2_STANDALONE_RELATIONAL_FP16.pt")
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s3a.get("stage")!="NRA2_S3A_REAL_43M_CPU_TRAIN32_LAST_LAYER_FEATURES_ONLY"
            or s3b.get("stage")!="NRA2_S3B_INDEPENDENT_HIDDEN320_GRAPH_ABI_PROXY_TRAIN24_HOLDOUT8"
            or s3b.get("candidate_model_accepted") is not False
            or s3b.get("output_gate")!=0.0
            or s3a.get("protected_584_digest_matches_S194_and_S2") is not True
            or standalone["fp16_checkpoint_sha256"]!=sha_file(learned)
            or s3a["standalone_NRA2_trained_FP16_sha256"]!=sha_file(learned)):
        raise AssertionError("S3C pinned frozen S194/S3A/S3B/NRA2 provenance drift")

    if (s3c.get("stage")!="NRA2_S3C_REAL_CPU_FROZEN_48_GRAPH_STRUCTURAL_PAIRS_ONLY"
            or s3c.get("all_616_token_digest_matches_S194") is not True
            or s3c.get("nra_output_gate")!=0.0
            or s3c.get("train_count")!=TRAIN_COUNT
            or s3c.get("novel_structural_heldout_count")!=HELD_COUNT):
        raise AssertionError("S3E requires immutable S3C corrected 616 J7 source")
    rows=build_graph_rows()
    if s3c["graph48_prompt_sha256_in_order"] != [r.prompt_sha for r in rows]:
        raise AssertionError("S3E S3C prompt order drift")
    assert len(rows)==TOTAL_COUNT and no_answer_template_lookup(rows)
    prompts=[r.question for r in rows]  # all model reads use TEXT only, no labels
    from calibri2.s112 import prompt_conditioned_first_token_mps as s112
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
    from v176.operators import clear

    original_load=s112.load_cpu_original
    original_probe=s190.probe_true_frozen_cpu
    handles=[]
    graph_prefills=0

    def load_frozen_with_read_ports(*args,**kwargs):
        cpu=original_load(*args,**kwargs)
        if handles:
            raise AssertionError("S3C CPU decoder must be one original immutable model")
        passive=NRA2PassiveFinalLayer(learned,standalone["fp16_checkpoint_sha256"]).install(cpu[0])
        collector=FrozenAllTokenFeatures().install(cpu[0])
        handles.append((passive,collector))
        return cpu

    def prefill_graphs_after_real_s190_probe(
        cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
        expansion,parent_old,head
    ):
        nonlocal graph_prefills
        # IMPORTANT: original S190 no-label 32 TRAIN + S194 capture runs first.
        original_probe(cpu,old,old_ids,routes,tok,bank,copy_net,bridge,full,
                       expansion,parent_old,head)
        if len(handles)!=1 or graph_prefills:
            raise AssertionError("S3C duplicate or missing native CPU model")
        model,sources,*_=cpu
        read,collector=handles[0]
        s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
        for q in prompts:
            # Match exact genuine S160 CPU S93 prefill phase and hidden ABI.
            text_vector=torch.as_tensor(s160.prompt_only_vector(tok,q),dtype=torch.float32)
            ids=s160.prompt_tokens(tok,q)
            with torch.inference_mode(),expansion.prefill_only(model),s160.prefill_only_tunnel(model,full):
                with s160.frozen_cpu_phase_scope(model,bridge,"S93",text_vector,phase="prefill_only"):
                    with collector.prompt_scope(q):
                        clear(sources)
                        prefill=model(input_ids=torch.tensor([ids],dtype=torch.long),
                                      past_key_values=None,use_cache=True,
                                      output_hidden_states=True)
                        if prefill.hidden_states[-1].shape[-1]!=320:
                            raise AssertionError("S3C real CPU hidden320 model ABI drift")
                        del prefill
                        clear(sources)
            graph_prefills+=1
        if len(collector.features)!=TOTAL_COUNT:
            raise AssertionError("S3C did not capture all new graph hidden320")

    # Never patch original J7 decoder. S194 remains sole 616 original-token capture.
    s112.load_cpu_original=load_frozen_with_read_ports
    s190.probe_true_frozen_cpu=prefill_graphs_after_real_s190_probe
    try:
        old,new,expected128,old_gold,novel,novel_gold,eos,s190_source=real_j7_baseline(
            argv,old_s87=Path(argv[-2]))
    finally:
        s112.load_cpu_original=original_load
        s190.probe_true_frozen_cpu=original_probe
        for passive,collector in handles:
            collector.uninstall()
            passive.uninstall()

    if graph_prefills!=TOTAL_COUNT or len(handles)!=1:
        raise AssertionError("S3C authentic extra frozen CPU graph prefills missing")
    passive,collector=handles[0]
    graph_sha=[r.prompt_sha for r in rows]
    features=collector.select(graph_sha)
    tokens,token_mask,token_lengths=collector.select_tokens(graph_sha)
    if (tokens.shape[0]!=TOTAL_COUNT or tokens.shape[-1]!=320
            or not torch.equal(tokens[torch.arange(TOTAL_COUNT),token_lengths.long()-1],features)
            or not torch.all(token_mask[torch.arange(TOTAL_COUNT),token_lengths.long()-1])):
        raise AssertionError("S3E exact last-layer token sequence does not match S3C-style last vector")
    if (features.shape!=(TOTAL_COUNT,320) or features.dtype!=torch.float16
            or collector.observed_tagged_calls!=TOTAL_COUNT
            or passive.report["output_gate"]!=0.
            or not passive.report["standalone_weights_frozen"]):
        raise AssertionError("S3C frozen 43M/NRA read-only graph hidden ABI failure")

    protected={**old,**new}
    if (len(old)!=456 or len(new)!=128 or len(novel)!=32
            or len(protected)!=584
            or sum(old[k]==old_gold[k] for k in old)!=456
            or sum(new[k]==expected128[k] for k in new)!=128
            or sum(novel[k]==novel_gold[k] for k in novel)!=0
            or output_digest(protected)!=s194["J7Retention584_frozen_baseline"]["protected_584_generation_digest"]
            or output_digest(novel)!=s194["J7Retention584_frozen_baseline"]["train32_generation_digest"]
            or s190_source["NEW_TRAIN_first_correct"]!=16):
        raise AssertionError("S3C entire original real 616 CPU decoder J7 changed")

    # Targets are created only after all 616 old CPU answers and new graph
    # CPU PREFILLS completed, never before/in neural feature extraction.
    targets=build_graph_targets(rows)
    original_feature=Path("inputs/nra2_stage3c/NRA2_S3C_REAL_43M_GRAPH48_HIDDEN320_FP16.safetensors")
    original_labels=Path("inputs/nra2_stage3c/NRA2_S3C_GRAPH48_STRUCTURAL_LABELS.safetensors")
    if (sha_file(original_feature)!=s3c["physical_feature_fp16_sha256"]
            or sha_file(original_labels)!=s3c["physical_graph_labels_sha256"]
            or not torch.equal(load_file(str(original_feature))["hidden320"],features)
            or any(not torch.equal(load_file(str(original_labels))[k],v)
                   for k,v in targets.items())):
        raise AssertionError("S3E new full-token acquisition changed original S3C 48 last states or labels")
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA2_S3C_REAL_43M_GRAPH48_HIDDEN320_FP16.safetensors"
    labels_file=out/"NRA2_S3C_GRAPH48_STRUCTURAL_LABELS.safetensors"
    save_file({"hidden320":features},str(feature_file))
    save_file(targets,str(labels_file))
    token_file=out/"NRA2_S3E_REAL_CPU_GRAPH48_ALL_TOKENS_FP16.safetensors"
    save_file({"token_hidden320":tokens,"token_mask":token_mask,
               "token_lengths":token_lengths},str(token_file))
    physical_tokens=load_file(str(token_file),device="cpu")
    if (not torch.equal(physical_tokens["token_hidden320"],tokens)
            or not torch.equal(physical_tokens["token_mask"],token_mask)
            or not torch.equal(physical_tokens["token_lengths"],token_lengths)):
        raise AssertionError("S3E physical token FP16 archive reload mismatch")
    a=load_file(str(feature_file),device="cpu")
    b=load_file(str(labels_file),device="cpu")
    if not torch.equal(a["hidden320"],features) or set(b)!=set(targets) or any(
        not torch.equal(b[k],v) for k,v in targets.items()
    ):
        raise AssertionError("Physical S3C real FP16 and graph label reload failed")
    report={
        "stage":"NRA2_S3E_REAL_CPU_FROZEN_FULL_TOKEN_GRAPH48_CAPTURE_ONLY",
        "original_corrected_S3C_run_id":43486292082,
        "full_token_feature_shape":list(tokens.shape),
        "full_token_sequence_lengths":token_lengths.tolist(),
        "token_mask_nonpadding_verified":True,
        "full_token_read_layer_index":11,
        "full_token_fp16_sha256":sha_file(token_file),
        "same_48_final_token_state_as_corrected_S3C":True,
        "S3D_unseen_exact_graphs":0,
        "S3D_unseen_reachability_correct":8,
        "real_native_CPU_43M_layer12_forward":True,
        "graph_prefill_count":TOTAL_COUNT,
        "real_hidden320_shape":[TOTAL_COUNT,320],
        "real_hidden320_dtype":"torch.float16",
        "graph_label_shapes":{k:list(v.shape) for k,v in targets.items()},
        "train_count":TRAIN_COUNT,
        "novel_structural_heldout_count":HELD_COUNT,
        "train_graph_active_nodes":[4,5],
        "heldout_graph_active_nodes":[6],
        "heldout_topologies_and_prompt_templates_disjoint":True,
        "train_positive":16,"heldout_positive":8,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "new32_full_EOS":0,
        "all_616_token_digest_matches_S194":True,
        "read_only_NRA":True,
        "calibri_and_green_tunnel_unchanged":True,
        "nra_output_gate":0.0,
        "graph_prefill_scoring":False,
        "train_graph_adapter_trained":False,
        "candidate_real_decoder_performance_gain_tested":False,
        "heldout_semantic_generalization_proven":False,
        "all_in_physical_bpp_of_candidate_tested":False,
        "paired_candidate_CPU_speed_tested":False,
        "candidate_model_accepted":False,
        "token_level_adapter_trained":False,
        "full_token_holdout_scored":False,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
        "physical_feature_fp16_sha256":sha_file(feature_file),
        "physical_graph_labels_sha256":sha_file(labels_file),
        "physical_reload_bitwise_equal":True,
        "graph48_prompt_sha256_in_order":graph_sha,
        "train_prompt_sha256":graph_sha[:TRAIN_COUNT],
        "heldout_prompt_sha256":graph_sha[TRAIN_COUNT:],
        "original_S194_checkpoint_sha256":s194["checkpoint_fingerprint_sha256"],
        "frozen_independent_NRA_sha256":sha_file(learned),
    }
    (out/"NRA2_S3C_GRAPH48_REAL_CPU_PAIRS_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("graph48_prompt_sha256_in_order",
                                   "train_prompt_sha256","heldout_prompt_sha256")},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
