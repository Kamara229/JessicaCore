"""NRA2 S3AH: read-only true 43M CPU [128,T,320] on precommitted S3AE EVAL ONLY, after S3AG FP16 locked.

Read-only; no fitting, threshold tuning or graph/scalar score. S194 616
original generated CPU token hashes must remain unchanged. Graph gold is
made only after CPU forward and 616 J7. Gate=0. LIVE, Render, Supabase,
APKs, main and original sealed QA/TEST remain untouched.
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
from nra2_bridge.stage3ae_train256_future_eval128_prereg import (
    build_rows,validate,commitment,TRAIN,EVAL,REVERSED_STYLES,
)
from nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets

GROUP_SIZES={EVAL:128}

TOTAL_COUNT=sum(GROUP_SIZES.values())
TRAIN_COUNT=0


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_targets(rows):
    """Labels created AFTER all true native 43M CPU graph prefills and original J7 616."""
    if len(rows)!=TOTAL_COUNT or any(r.split!=EVAL for r in rows):
        raise AssertionError("S3AH materialize labels for EVAL128 only after original native CPU")
    edge=torch.zeros((TOTAL_COUNT,6,6),dtype=torch.float16)
    src=torch.zeros((TOTAL_COUNT,6),dtype=torch.float16)
    dst=torch.zeros_like(src)
    reach=torch.zeros((TOTAL_COUNT,),dtype=torch.int32)
    for i,r in enumerate(rows):
        for a,b in r.edges:
            edge[i,a,b]=1.
        src[i,r.src]=1.
        dst[i,r.dst]=1.
        reach[i]=int(r.reachable)
    return {"adjacency":edge,"source":src,"target":dst,"reachability":reach}


def main():
    if len(sys.argv)<15:
        raise SystemExit("S3AH needs original S194 616 true native CPU J7 CLI")
    argv=list(sys.argv)
    out=Path(argv[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3AH evidence must not be overwritten")
    s194=json.loads(Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").read_text())
    s3e=json.loads(Path("inputs/nra2_stage3e/NRA2_S3E_FULL_TOKEN_CPU_GRAPH48_AUDIT.json").read_text())
    s3a=json.loads(Path("inputs/nra2_stage3a/NRA2_S3A_REAL_CPU_32_TRAIN_FEATURE_AUDIT.json").read_text())
    s3b=json.loads(Path("inputs/nra2_stage3b/NRA2_S3B_24TRAIN_8HELDOUT_PROXY_AUDIT.json").read_text())
    standalone=json.loads(Path("inputs/nra2/NRA2_STANDALONE_REPORT.json").read_text())
    learned=Path("inputs/nra2/NRA2_STANDALONE_RELATIONAL_FP16.pt")
    old_tokens=Path("inputs/nra2_stage3e/NRA2_S3E_REAL_CPU_GRAPH48_ALL_TOKENS_FP16.safetensors")
    old_hidden=Path("inputs/nra2_stage3e/NRA2_S3E_REAL_43M_GRAPH48_HIDDEN320_FP16.safetensors")
    old_labels=Path("inputs/nra2_stage3e/NRA2_S3E_GRAPH48_STRUCTURAL_LABELS.safetensors")
    earlier=json.loads(Path("inputs/nra2_stage3h/NRA2_S3H_128_FACTORIAL_REAL_CPU_AUDIT.json").read_text())
    earlier_token_file=Path("inputs/nra2_stage3h/NRA2_S3H_REAL_CPU_GRAPH128_ALL_TOKENS_FP16.safetensors")
    if (earlier.get("stage")!="NRA2_S3H_REAL_CPU_FROZEN_128_FACTORIAL_TOKEN_CAPTURE_ONLY"
            or earlier.get("all_616_token_digest_matches_S194") is not True
            or earlier.get("full_token_fp16_sha256")!=sha_file(earlier_token_file)
            or earlier.get("nra_output_gate")!=0.0
            or earlier.get("physical_reload_bitwise_equal") is not True
            or earlier.get("train_count")!=64
            or earlier.get("graph_prefill_count")!=128):
        raise AssertionError("S3AH frozen S3H authentic 128-token sequence and J7 drift")
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s3e.get("stage")!="NRA2_S3E_REAL_CPU_FROZEN_FULL_TOKEN_GRAPH48_CAPTURE_ONLY"
            or s3e.get("original_456_full_EOS")!=456
            or s3e.get("original_new128_full_EOS")!=128
            or s3e.get("all_616_token_digest_matches_S194") is not True
            or s3e.get("physical_reload_bitwise_equal") is not True
            or s3e.get("nra_output_gate")!=0.
            or s3e.get("cross_run_S3C_within_two_adjacent_fp16_steps") is not True
            or s3e.get("cross_run_S3C_labels_bitwise_equal") is not True
            or s3e.get("full_token_fp16_sha256")!=sha_file(old_tokens)
            or s3e.get("physical_feature_fp16_sha256")!=sha_file(old_hidden)
            or s3e.get("physical_graph_labels_sha256")!=sha_file(old_labels)
            or s3a.get("stage")!="NRA2_S3A_REAL_43M_CPU_TRAIN32_LAST_LAYER_FEATURES_ONLY"
            or s3a.get("protected_584_digest_matches_S194_and_S2") is not True
            or s3b.get("stage")!="NRA2_S3B_INDEPENDENT_HIDDEN320_GRAPH_ABI_PROXY_TRAIN24_HOLDOUT8"
            or s3b.get("output_gate")!=0.
            or s3b.get("candidate_model_accepted") is not False
            or standalone.get("fp16_checkpoint_sha256")!=sha_file(learned)):
        raise AssertionError("S3AH immutable S194/S3E/S3A/S3B/NRA provenance invalid")
    # Training has ended, S3AG checkpoint is physically fixed. Here EVAL128
    # undergoes original frozen 43M CPU prefill ONLY; no candidate scoring.
    # EVAL128 must never be used for optimizer updates or future tuning.
    # Lock and SHA-check the physical S3AG FP16 artifact before any EVAL CPU.
    s3ag=json.loads(Path("inputs/nra2_stage3ag/NRA2_S3AG_TRAIN512_ROLE_HEAD_AUDIT.json").read_text())
    s3ag_weights=Path("inputs/nra2_stage3ag/NRA2_S3AG_FROZEN_GRAPH_ROLE512_TRAIN_FP16.safetensors")
    if (s3ag.get("stage")!="NRA2_S3AG_TRAIN512_FROZEN_S3O_GRAPH_S3Z_ROLE_ONLY"
        or s3ag.get("train_count")!=512
        or s3ag.get("new_S3AG_physical_fp16_sha256")!="2c23de87efc2369ed6fc0880604420070c116a4ee70b66a4eb0a946f1afeb118"
        or s3ag["new_S3AG_physical_fp16_sha256"]!=sha_file(s3ag_weights)
        or s3ag.get("future_independent_S3AE_EVAL128_never_prefilled_scored_or_trained") is not True
        or s3ag.get("original_graph_physical_fp16_bitwise_equal") is not True
        or s3ag.get("full_S194_original_616_J7_digest_preserved_in_S3N_S3Y_S3AF") is not True
        or s3ag.get("candidate_accepted") is not False
        or s3ag.get("output_gate")!=0.0):
        raise AssertionError("S3AH prior S3AG physically locked, approved TRAIN512, future EVAL pristine")
    precommitted_rows=build_rows()
    if len(precommitted_rows)!=384 or not validate(precommitted_rows):
        raise AssertionError("S3AH pinned S3AE 256 TRAIN + 128 FUTURE EVAL prereg drift")
    rows=[r for r in precommitted_rows if r.split==EVAL]
    if len(rows)!=TOTAL_COUNT or len({r.topology_key for r in rows})!=32:
        raise AssertionError("S3AH EVAL-only 128 distinct SHA and 32 new graph topologies required")
    prereg_path=Path("inputs/nra2_stage3ae/NRA2_S3AE_NEW_TRAIN256_FUTURE_EVAL128_PREREG.json")
    prior=json.loads(prereg_path.read_text())
    immutable=commitment(precommitted_rows)
    if (prior!=immutable
        or prior.get("stage")!="NRA2_S3AE_PRECOMMIT_DISJOINT_NEW_TRAIN256_FUTURE_EVAL128_NO_SCORE"
        or prior.get("commitment_sha256")!="0ccbb2b19b3feb93b79f5dcbc892f2e414a564b6e4d83e52997098346e4f480f"
        or prior.get("original_S3AC_successful_run")!=43535868240
        or prior.get("original_S3AD_post_score_diagnosis_successful_run")!=43536390826
        or prior.get("previous_S3AA_is_already_scored_and_NOT_future_blind") is not True
        or prior.get("new_eval_excluded_from_optimizer_and_checkpoint_selection") is not True
        or prior.get("future_CPU_acquisition_and_training_not_started") is not True
        or prior.get("no_weights_or_training_or_scores") is not True
        or prior.get("original_S194_616_J7_guard_required") is not True
        or prior.get("no_candidate_accepted") is not True
        or prior.get("gate")!=0.
        or prior.get("splits",{}).get(TRAIN,{}).get("count")!=256
        or prior["splits"][TRAIN]["yes_count"]!=128
        or prior["splits"][TRAIN]["destination_first_count"]!=128
        or prior["splits"][TRAIN]["question_sha256"]!=[r.prompt_sha256 for r in rows]
        or prior["splits"][TRAIN]["source_order_query_offsets"]!=[query_char_offsets(r.question) for r in rows]
        or prior.get("splits",{}).get(EVAL,{}).get("count")!=128
        or prior["splits"][EVAL]["topology_count"]!=32
        or prior["splits"][EVAL]["yes_count"]!=64
        or prior["splits"][EVAL]["destination_first_count"]!=64):
        raise AssertionError("S3AH pinned TRAIN256 prereg / preserved future EVAL128 SHA drift")
    previously=json.loads(Path("inputs/nra2_stage3n/NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    old_s3n_tokens=Path("inputs/nra2_stage3n/NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors")
    if (previously.get("stage")!="NRA2_S3N_REAL_CPU_FROZEN_192_ROLEBALANCED_TOKEN_CAPTURE_ONLY"
        or previously.get("all_616_token_digest_matches_S194") is not True
        or previously.get("full_token_fp16_sha256")!=sha_file(old_s3n_tokens)
        or previously.get("physical_reload_bitwise_equal") is not True):
        raise AssertionError("S3AH original S3N 192 native CPU immutable evidence invalid")
    # From this point until the original CPU work completes ONLY raw text
    # prompts are used. Gold graph labels never feed a hook or the model.
    prompts=[r.question for r in rows]
    prompt_hashes={hashlib.sha256(q.encode("utf-8")).hexdigest() for q in prompts}
    previous_sha=set(s3e["graph48_prompt_sha256_in_order"])
    previous_sha.update(
        prompt_sha for group in earlier["split_groups"].values()
        for prompt_sha in group["prompt_sha256"]
    )
    previous_sha.update(
        h for v in previously["split_groups"].values() for h in v["prompt_sha256"]
    )
    # Original S3AF TRAIN256 was captured before S3AG training; future
    # independent EVAL128 must share no exact prompt with this group.
    train_capture=json.loads(Path("inputs/nra2_stage3af/NRA2_S3AF_TRAIN256_REAL_CPU_J7_AUDIT.json").read_text())
    previous_sha.update(train_capture["split_groups"][TRAIN]["prompt_sha256"])
    if prompt_hashes & previous_sha:
        raise AssertionError("S3AH independent EVAL128 prompt overlaps previously captured CPU source")
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
            raise AssertionError("S3AH CPU decoder must be one original immutable model")
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
            raise AssertionError("S3AH duplicate or missing native CPU model")
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
                            raise AssertionError("S3AH real CPU hidden320 model ABI drift")
                        del prefill
                        clear(sources)
            graph_prefills+=1
        if len(collector.features)!=TOTAL_COUNT:
            raise AssertionError("S3AH did not capture all independent EVAL128 hidden320")

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
        raise AssertionError("S3AH authentic extra frozen CPU graph prefills missing")
    passive,collector=handles[0]
    graph_sha=[r.prompt_sha256 for r in rows]
    features=collector.select(graph_sha)
    tokens,token_mask,token_lengths=collector.select_tokens(graph_sha)
    if (tokens.shape[0]!=TOTAL_COUNT or tokens.shape[-1]!=320
            or not torch.equal(tokens[torch.arange(TOTAL_COUNT),token_lengths.long()-1],features)
            or not torch.all(token_mask[torch.arange(TOTAL_COUNT),token_lengths.long()-1])):
        raise AssertionError("S3AH exact last-layer token sequence does not match S3AH-style last vector")
    if (features.shape!=(TOTAL_COUNT,320) or features.dtype!=torch.float16
            or collector.observed_tagged_calls!=TOTAL_COUNT
            or passive.report["output_gate"]!=0.
            or not passive.report["standalone_weights_frozen"]):
        raise AssertionError("S3AH frozen 43M/NRA read-only graph hidden ABI failure")

    protected={**old,**new}
    if (len(old)!=456 or len(new)!=128 or len(novel)!=32
            or len(protected)!=584
            or sum(old[k]==old_gold[k] for k in old)!=456
            or sum(new[k]==expected128[k] for k in new)!=128
            or sum(novel[k]==novel_gold[k] for k in novel)!=0
            or output_digest(protected)!=s194["J7Retention584_frozen_baseline"]["protected_584_generation_digest"]
            or output_digest(novel)!=s194["J7Retention584_frozen_baseline"]["train32_generation_digest"]
            or s190_source["NEW_TRAIN_first_correct"]!=16):
        raise AssertionError("S3AH entire original real 616 CPU decoder J7 changed")


    # Novel grammar: exact S160 tokenizer and S3O query mask
    # only after original 616 J7 digests, without gold or model scores.
    from research.nra2_bridge.stage3k_train_real_entity_binding import exact_original_tokenizer
    from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
    from research.nra2_bridge.stage3o_neural_query_roles import build_query_text_mask
    native_tokenizer=exact_original_tokenizer()
    aligned=validate_without_gold(native_tokenizer,prompts)
    if any(a["tokens"]!=int(token_lengths[i]) for i,a in enumerate(aligned)):
        raise AssertionError("S3AH native S160 tokenizer length differs from CPU")
    qmask=build_query_text_mask(native_tokenizer,prompts,aligned,int(tokens.shape[1]))
    if (qmask.query_mask.shape!=(TOTAL_COUNT,int(tokens.shape[1]))
        or qmask.node_order.shape!=(TOTAL_COUNT,2)
        or not torch.equal(qmask.node_order,torch.tensor(
            [query_char_offsets(q)["text_order_nodes"] for q in prompts],dtype=torch.long))):
        raise AssertionError("S3AH native tokenizer S3O novel query mention ABI drift")
    # Gold labels are created only after real CPU, J7 and query-token alignment.
    targets=make_targets(rows)
    out.mkdir(parents=True,exist_ok=True)
    feature_file=out/"NRA2_S3AH_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    labels_file=out/"NRA2_S3AH_EVAL128_STRUCTURAL_LABELS.safetensors"
    token_file=out/"NRA2_S3AH_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    save_file({"hidden320":features.contiguous()},str(feature_file))
    save_file({k:v.contiguous() for k,v in targets.items()},str(labels_file))
    save_file({"token_hidden320":tokens.contiguous(),"token_mask":token_mask.contiguous(),
               "token_lengths":token_lengths.contiguous()},str(token_file))
    p=load_file(str(token_file),device="cpu")
    f=load_file(str(feature_file),device="cpu")["hidden320"]
    lab=load_file(str(labels_file),device="cpu")
    if (not torch.equal(p["token_hidden320"],tokens)
            or not torch.equal(p["token_mask"],token_mask)
            or not torch.equal(p["token_lengths"],token_lengths)
            or not torch.equal(f,features)
            or not torch.equal(p["token_hidden320"][
                torch.arange(TOTAL_COUNT),p["token_lengths"].long()-1],f)
            or any(not torch.equal(lab[k],v) for k,v in targets.items())):
        raise AssertionError("S3AH FP16 physical reload, mask, or labels failed")
    grouped={}
    offset=0
    for name,count in GROUP_SIZES.items():
        group_rows=rows[offset:offset+count]
        grouped[name]={
            "start":offset,"count":count,
            "prompt_sha256":[r.prompt_sha256 for r in group_rows],
            "positive":sum(r.reachable for r in group_rows),
            "node_lengths":sorted({r.active_nodes for r in group_rows})
        }
        offset+=count
    if offset!=TOTAL_COUNT or len(set(graph_sha))!=TOTAL_COUNT:
        raise AssertionError("S3AH split indexing or prompt uniqueness invalid")
    report={
        "stage":"NRA2_S3AH_FROZEN_S3AE_INDEPENDENT_EVAL128_NO_SCORE_CPU_CAPTURE_ONLY",
        "S3AE_registered_commitment_sha256":prior["commitment_sha256"],
        "S3AE_successful_prereg_run":43537386173,
        "S3AG_successful_frozen_adapter_run":43539383181,
        "S3AG_frozen_pre_EVAL_checkpoint_sha256":sha_file(s3ag_weights),
        "S3AE_EVAL128_source_sha256":hashlib.sha256(json.dumps(
            prior["splits"][EVAL]["question_sha256"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "S3AE_EVAL128_query_char_offsets_sha256":hashlib.sha256(json.dumps(
            prior["splits"][EVAL]["source_order_query_offsets"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "S3AE_future_EVAL128_commitment_sha256":hashlib.sha256(json.dumps(
            prior["splits"][EVAL]["question_sha256"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "S3AE_EVAL128_prefilled_after_S3AG_locked":True,
        "S3AE_EVAL128_not_scored_or_trained":True,
        "S3AE_future_EVAL128_graph_topologies":32,
        "S3AE_EVAL128_graph_topologies":32,
        "S3AE_EVAL128_yes":64,
        "S3AE_EVAL128_no":64,
        "S3AE_EVAL128_source_first":64,
        "S3AE_EVAL128_destination_first":64,
        "native_tokenizer_S3O_query_alignment_verified":True,
        "S3N_native_CPU_source_run":43522555326,
        "S3N_192_source_token_fp16_sha256":sha_file(old_s3n_tokens),
        "S3E_successful_parent_run":43512049506,
        "S3E_prior_token_artifact_sha256":sha_file(old_tokens),
        "S3H_successful_CPU_run":43517070838,
        "S3H_prior_token_artifact_sha256":sha_file(earlier_token_file),
        "S194_original_checkpoint_sha256":s194["checkpoint_fingerprint_sha256"],
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "original_new32_full_EOS":0,
        "all_616_token_digest_matches_S194":True,
        "real_native_CPU_43M_layer12_forward":True,
        "full_token_read_layer_index":11,
        "graph_prefill_count":TOTAL_COUNT,
        "train_count":TRAIN_COUNT,
        "frozen_soft_and_hard_candidate_scored":False,
        "predeclared_once_only_EVAL128":True,
        "no_further_training_or_tuning_using_EVAL128":True,
        "group_sizes":dict(GROUP_SIZES),
        "split_groups":grouped,
        "real_hidden320_shape":list(features.shape),
        "full_token_feature_shape":list(tokens.shape),
        "token_lengths":token_lengths.tolist(),
        "token_mask_nonpadding_verified":True,
        "full_token_fp16_sha256":sha_file(token_file),
        "physical_feature_fp16_sha256":sha_file(feature_file),
        "physical_graph_labels_sha256":sha_file(labels_file),
        "frozen_independent_NRA_sha256":sha_file(learned),
        "read_only_NRA":True,"calibri_and_green_tunnel_unchanged":True,
        "nra_output_gate":0.0,
        "physical_reload_bitwise_equal":True,
        "labels_materialized_after_native_CPU":True,
        "control_groups_scored":False,
        "adapter_trained":False,
        "candidate_accepted":False,
        "full_EOS_candidate_gain_tested":False,
        "all_in_physical_bpp_tested":False,
        "paired_candidate_CPU_speed_tested":False,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
    }
    (out/"NRA2_S3AH_EVAL128_REAL_CPU_J7_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("split_groups","token_lengths")},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
