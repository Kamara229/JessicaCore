"""S49 physical original43M+Green Tunnel CPU frozen word338 sidecar smoke.

Downloads the *genuine* SHA-locked S22c REAL frozen Calibri43M+GT
layer8 word338 capture from earlier 12-layer CPU S22b, derived after
verified original 616J7, but DOES NOT reload 43M original model, rerun J7
or inject quotes into any transformer layer. That would need a separate
S50 native-runtime capture/retention experiment. This stage checks real
native ABI compatibility, exact identity with gate0 and the presence of
word320 vs quote raw feature in an isolated 339-D adapter workspace.

ONLY S22c raw338 + manifest are downloaded; graph gold omitted entirely.
S47 and S38 adversarial sources are not imported. 616 J7 unmodified by
construction, NOT rerun as part of this script. No model gain claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    text_only_inputs,manifest as source_manifest,FROZEN_S22_TRAIN256_SHA256,
)
from research.nra3_s20.s49_real43m_gt_quote_sidecar_zero_gate import (
    Frozen43MGreenTunnelQuotePort,frozen_native_manifest_digest,
)

STAGE="NRA3_S49_REAL_ORIGINAL43M_GT_ARCHIVED_LAYER8_NATIVE_256_ZERO_GATE_S48_QUOTE_SIDE_CAR"
PARENT_RUN=44191302703
S22B_CPU_RUN=44189739539
S48_RUN=44259167542
S22C_STAGE="NRA3_S22C_REAL_ORIGINAL43M_GT_NEW256_SEPARATE_GRAPH_TRAIN_SHARDS"


def require(ok,what):
    if not bool(ok):
        raise ValueError("S49 original real43M+GT quote sidecar: "+what)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    require(not args.out.exists(),"research report overwrite refused")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(args.features.is_file() and args.manifest.is_file(),
            "missing original 43M/GT real capture")
    m=json.loads(args.manifest.read_text(encoding="utf-8"))
    require(m.get("stage")==S22C_STAGE
            and m.get("S22_sha_locked_synthetic_TRAIN256_graph_gold")
                ==FROZEN_S22_TRAIN256_SHA256
            and m.get("S22_original_protected_616_J7_parent_unchanged") is True
            and m.get("256_curated_synthetic_train_texts")==256
            and m.get("genuine_original_CPU_hidden_plus_plain_raw18_ONLY_as_model_inputs") is True
            and m.get("human_graph_targets_NEVER_passed_to_original_or_adapter_forward") is True
            and m.get("no_consumed_EVAL32_EVAL128_S17c_S15c_S12c") is True
            and m.get("new_active_candidate_616_J7_CPU_bpp_NOT_CERTIFIED") is True
            and m.get("raw338_word_features_filename")==args.features.name
            and m.get("raw338_word_features_SHA256")==digest(args.features)
            and m.get("S22b_original43M_GT_CPU_hidden320_FP16_sha")
            and not m.get("candidate_accepted")
            and not m.get("activation_allowed")
            and m.get("output_gate")==0,
            "S22c real frozen Calibri43M+GT source manifest drift")
    t=text_only_inputs()
    require(len(t)==256 and len(set(t))==256
            and source_manifest()["source_sha256"]==FROZEN_S22_TRAIN256_SHA256,
            "S22 SHA-sealed original TRAIN256 prompt order changed")
    physical=load_file(str(args.features),device="cpu")
    require(set(physical)=={"raw_text_frozen_word338","valid_word_mask"},
            "gold file must NOT be downloaded or present in the native features")
    x=physical["raw_text_frozen_word338"].detach().contiguous()
    valid=physical["valid_word_mask"].detach().contiguous()
    require(x.ndim==3 and x.shape[0]==256 and x.shape[-1]==338
            and x.dtype==torch.float32 and valid.shape==x.shape[:2]
            and valid.dtype==torch.bool,
            "real original43M layer8 post-GT hidden320+raw18 ABI drift")
    orig_sha=frozen_native_manifest_digest(x,valid)
    orig_hidden_sha=hashlib.sha256(x[...,:320].contiguous().numpy().tobytes()).hexdigest()
    orig_raw18_sha=hashlib.sha256(x[...,320:].contiguous().numpy().tobytes()).hexdigest()
    port=Frozen43MGreenTunnelQuotePort()
    out=port.forward_original_unchanged(x,valid,t)
    require(out is x and out.data_ptr()==x.data_ptr()
            and torch.equal(x,out) and out.shape==x.shape,
            "gate0 returned a modified original 43M feature tensor")
    appended,sidecar=port.shadow_augmented_features_only(x,valid,t)
    require(appended.shape==(256,x.shape[1],339)
            and torch.equal(appended[...,:338],x)
            and torch.equal(appended[...,:320],x[...,:320])
            and torch.equal(appended[...,320:338],x[...,320:])
            and bool((appended[~valid]==0).all())
            and appended.data_ptr()!=x.data_ptr(),
            "disconnected append-only S48 quote channel changed 43M GT source")
    require(frozen_native_manifest_digest(x,valid)==orig_sha
            and hashlib.sha256(x[...,:320].contiguous().numpy().tobytes()).hexdigest()==orig_hidden_sha
            and hashlib.sha256(x[...,320:].contiguous().numpy().tobytes()).hexdigest()==orig_raw18_sha,
            "native original 43M CPU tensor changed after S48 observer")
    require(bool(torch.isfinite(appended).all())
            and len(sidecar.per_row_tokens)==256
            and len(sidecar.per_row_quote_tokens)==256,
            "S49 original43M quote sidecar malformed")
    candidate_probe_pass=False
    try:
        port.output_gate=1
        port.forward_original_unchanged(x,valid,t)
    except ValueError:
        candidate_probe_pass=True
    finally:
        port.output_gate=0
    require(candidate_probe_pass,"S49 forbidden candidate gate was not rejected")
    require(port.output_gate==0 and port.read_only
            and not port.candidate_accepted and not port.activation_allowed
            and not port.training_allowed,
            "S49 finished with enabled or trainable sidecar")
    quote_positive=int(sidecar.text_only_quote_mask.sum().item())
    supported=int(sidecar.supported_mask.sum().item())
    actual_native_words=int(valid.sum().item())
    info={
        "stage":STAGE,
        "original_TRUE43M_12layer_GreenTunnel_native_S22b_CPU_parent_run":S22B_CPU_RUN,
        "original_TRUE43M_GT_SHA_sealed_S22c_raw338_parent_run":PARENT_RUN,
        "S48_typographic_quote_graph_prior_TRAIN36_HELD26_CPU_run":S48_RUN,
        "original_real43M_source_train256_canonical_sha256":FROZEN_S22_TRAIN256_SHA256,
        "original_real43M_GT_native_word338_safetensors_sha256":digest(args.features),
        "original_real43M_S22c_source_manifest_sha256":digest(args.manifest),
        "original_S22b_CPU_layer8_hidden320_FP16_parent_sha256":m["S22b_original43M_GT_CPU_hidden320_FP16_sha"],
        "original_REAL43M_word338_tensor_sha256_before_after":orig_sha,
        "original_REAL43M_layer8_hidden320_tensor_sha256_before_after":orig_hidden_sha,
        "original_REAL43M_raw18_tensor_sha256_before_after":orig_raw18_sha,
        "true_original_CPU_12layers_GT_verified_BY_S22b_parent_not_rerun_here":True,
        "original_protected_616_J7_verified_BY_S22b_parent_NOT_rerun_in_S49":True,
        "new_original43m_616J7_retention_did_NOT_run":True,
        "original43M_weights_checkpoint_NOT_loaded_nor_modified":True,
        "original43M_runtime_hook_NOT_installed":True,
        "original_TRUE43M_word_vectors_256_synthetic_TRAIN_rows":256,
        "original_TRUE43M_word_vectors_total":actual_native_words,
        "original_TRUE43M_word_width":int(x.shape[1]),
        "rows_S48_quote_parser_supported":256-len(sidecar.unsupported_row_ids),
        "rows_quote_parser_unsupported_fail_closed":list(sidecar.unsupported_row_ids),
        "S48_quote_words_in_sealed_S22_SYNTHETIC_TRAIN256":quote_positive,
        "S48_quote_pair_completed_count":sum(sidecar.per_row_matched_pairs),
        "S48_unmatched_outer_opener_count":sum(sidecar.per_row_unmatched_openers),
        "quote_supported_word_count":supported,
        "original338_output_same_Python_object_AND_data_ptr":True,
        "original_hidden320_and_raw18_bitwise_and_SHA_unchanged":True,
        "research_only_339D_augmented_adapter_channel_separate_not_injected_into_model":True,
        "nonzero_quote_gate_explicitly_rejected":True,
        "learned_S48_four_parameter_shadow_NOT_assumed_compatible_with_REAL43M":True,
        "original43M_post_GT_12layer_tensor_only_READ_after_prior_replay":True,
        "no_S22c_graph_gold_file_downloaded_or_loaded":True,
        "no_gold_in_quote_parser_or_sidecar_forward":True,
        "original_S38_protected32_not_accessed":True,
        "previously_examined_S47_challenge36_not_accessed_or_tuned":True,
        "no_S48_HELD26_scored_or_tuned":True,
        "no_independent_EVAL32_or_EVAL128_used":True,
        "616J7_bpp_CB59_active_candidate_not_measured":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/"NRA3_S49_REAL43M_GT_S48_QUOTE_ZERO_GATE_NATIVE_LAYER8_BRIDGE.json"
    path.write_text(json.dumps(info,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                    encoding="utf-8")
    print("S49_REAL43M_GT_NATIVE_LAYER8_ACTUAL_CPU_ARCHIVE",json.dumps({
        "real_native_train256_words":actual_native_words,
        "s48_quote_tokens":quote_positive,
        "s48_completed_quote_pairs":sum(sidecar.per_row_matched_pairs),
        "unsupported_rows":len(sidecar.unsupported_row_ids),
        "unchanged_orig_sha256":orig_sha,
        "output_gate":0,
        "native_43M_NOT_rerun":True,
    },sort_keys=True),flush=True)
    print("S49_ZERO_GATE_TRUE43M_12LAYER_GT_SIDE_CAR_ONLY_NO_EVAL_NO_DEPLOY",flush=True)

if __name__=="__main__":
    main()
