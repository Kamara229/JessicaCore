"""NRA2 Stage3A: frozen genuine 43M CPU hidden representations for S190 TRAIN.

Run the entire original 616 S194 real generated-token J7, preserving the exact
output digests. During S160 real CPU cached prefill of the 32 S190 declared
NEW TRAIN questions, observe the actual *post-layer-12* 320D hidden feature
(no model modification). Capture BEFORE gold answers are consulted.

Outputs are physical FP16 feature tensors and SHA-indexed order only.
They are TRAIN-only research inputs for later independent feature→relations
adapter training, NOT an integrated/admitted neural candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file, load_file

from calibri2.s194.actual_cpu_j7_bridge import real_j7_baseline, output_digest
from nra2_bridge.frozen_features import FrozenFinalFeatures, prompt_sha
from nra2_bridge.passive_final_layer import NRA2PassiveFinalLayer


def sha_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    if len(sys.argv)<15:
        raise SystemExit("NRA2 Stage3A requires the full immutable S194 J7 CPU CLI")
    args=list(sys.argv)
    out=Path(args[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("NRA2 Stage3A output must be new and immutable")

    s194_file=Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json")
    s2_file=Path("inputs/nra2_stage2/NRA2_REAL_43M_PASSIVE_616_J7.json")
    learned_file=Path("inputs/nra2/NRA2_STANDALONE_RELATIONAL_FP16.pt")
    nra_training=Path("inputs/nra2/NRA2_STANDALONE_REPORT.json")
    if not all(p.is_file() for p in (s194_file,s2_file,learned_file,nra_training)):
        raise AssertionError("Missing physical S194, Stage2 or independent NRA2 source")
    s194=json.loads(s194_file.read_text())
    s2=json.loads(s2_file.read_text())
    nra=json.loads(nra_training.read_text())
    if (s194.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or s2.get("stage")!="NRA2_CALIBRI_FINAL_LAYER_READ_ONLY"
            or s2.get("model_accepted") is not False
            or s2.get("exact_584plus32_real_generated_token_ID_digest_matches_S194") is not True
            or s2.get("J7_baseline_full_retention") is not True
            or s2.get("nra2_production_accepted") is not False
            or nra.get("stage")!="NRA2_STANDALONE_RELATIONAL_V0"
            or nra.get("candidate_status")!="EXPLORATORY_STANDALONE_PASS_NOT_INTEGRATION"
            or nra.get("calibri_2_integrated") is not False
            or s2["standalone_fp16_sha256"]!=nra["fp16_checkpoint_sha256"]):
        raise AssertionError("Frozen NRA2 Stage2 trained/CPU J7 certificate drift")

    # Install two *read-only* hooks at the SAME single terminal layer, in a
    # single CPU process. The existing S194 full-616 generated-token interceptor
    # remains the only source of predictions. No new decoder/router is introduced.
    from calibri2.s112 import prompt_conditioned_first_token_mps as s112
    from calibri2.s160 import real_parallel_slots_j7 as s160

    original_load=s112.load_cpu_original
    original_decoder=s160.cached_full_cpu_parallel
    handles=[]

    def load_with_feature_tap(*pos,**kwargs):
        cpu=original_load(*pos,**kwargs)
        if handles:
            raise AssertionError("NRA2 Stage3A must observe exactly one authentic CPU core")
        read=NRA2PassiveFinalLayer(learned_file,nra["fp16_checkpoint_sha256"])
        read.install(cpu[0])
        feature=FrozenFinalFeatures().install(cpu[0])
        handles.append((read,feature))
        return cpu

    def tag_real_cpu_decoding(*pos,**kwargs):
        question=pos[6] if len(pos)>6 else kwargs.get("question")
        if not handles:
            raise AssertionError("NRA2 Stage3A real CPU decoder has no installed final-layer tap")
        feature=handles[0][1]
        with feature.prompt_scope(question):
            return original_decoder(*pos,**kwargs)

    # CRITICAL ORDER: hooks must be patched before S194 lazily loads S190
    # and transitively S168, which freezes a reference to the CPU decoder.
    s112.load_cpu_original=load_with_feature_tap
    s160.cached_full_cpu_parallel=tag_real_cpu_decoding
    try:
        old,new,targets,old_gold,novel,novel_gold,eos,source=real_j7_baseline(
            args,old_s87=Path(args[-2]))
    finally:
        s112.load_cpu_original=original_load
        s160.cached_full_cpu_parallel=original_decoder
        for passive,feature in handles:
            feature.uninstall()
            passive.uninstall()

    if len(handles)!=1:
        raise AssertionError("NRA2 Stage3A genuine 43M frozen model was not loaded")
    passive,feature=handles[0]
    from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
    rows=new_train_paraphrase_probes()
    ordered=sorted(prompt_sha(q) for q,_,_ in rows)
    if len(ordered)!=32:
        raise AssertionError("NRA2 Stage3A requires exactly 32 S190 TRAIN prompts")
    features=feature.select(ordered)

    # True old+new+novel J7 generation must be identical to S194 / Stage2.
    original_all=dict(old)
    if set(original_all)&set(new) or (set(original_all)|set(new))&set(novel):
        raise AssertionError("NRA2 Stage3A original hashed TRAIN identity collision")
    original_all.update(new)
    protected_digest=output_digest(original_all)
    novel_digest=output_digest(novel)
    if (protected_digest!=s194["J7Retention584_frozen_baseline"]["protected_584_generation_digest"]
            or novel_digest!=s194["J7Retention584_frozen_baseline"]["train32_generation_digest"]
            or protected_digest!=s2["protected_generation_sha256"]
            or novel_digest!=s2["new32_generation_sha256"]):
        raise AssertionError("NRA2 Stage3A altered 616 original real generated token sequences")
    if (len(old)!=456 or len(new)!=128 or len(novel)!=32
            or sum(old[k]==old_gold[k] for k in old)!=456
            or sum(new[k]==targets[k] for k in new)!=128
            or sum(novel[k]==novel_gold[k] for k in novel)!=0
            or sum(novel[k][0]==novel_gold[k][0] for k in novel)!=16):
        raise AssertionError("NRA2 Stage3A original real frozen 616 J7 correctness drift")
    if (passive.report["output_gate"]!=0.0 or not passive.report["standalone_weights_frozen"]
            or passive.report["real_cpu_forward_calls"]<=0
            or feature.observed_tagged_calls<32):
        raise AssertionError("NRA2 Stage3A did not physically observe real decoder prefill")

    # Enforce bounded, completely frozen train-only representations. Store no
    # labels, gold target IDs, raw prompts, user chats or QA/TEST examples.
    out.mkdir(parents=True,exist_ok=True)
    tensor_file=out/"NRA2_S3A_REAL_CPU_32_TRAIN_HIDDEN320_FP16.safetensors"
    save_file({"final_hidden_320":features},str(tensor_file))
    reread=load_file(str(tensor_file),device="cpu")["final_hidden_320"]
    if not torch.equal(features,reread):
        raise AssertionError("Physical FP16 save/reload corrupted real CPU hidden states")

    metadata={
        "stage":"NRA2_S3A_REAL_43M_CPU_TRAIN32_LAST_LAYER_FEATURES_ONLY",
        "original_s194_physical_checkpoint_sha256":s194["checkpoint_fingerprint_sha256"],
        "S194_actual_real_CPU_616_reexecuted":True,
        "standalone_NRA2_trained_FP16_sha256":nra["fp16_checkpoint_sha256"],
        "NRA2_area_loaded_and_frozen":True,
        "data_family":"S190 immutable new TRAIN paraphrase 32 only",
        "dataset_question_sha256_sorted":ordered,
        "source_prompt_texts_exported":False,
        "target_answers_or_expected_token_IDs_in_feature_file":False,
        "hidden_features_collected_before_gold_scoring":True,
        "feature_tensor":"final_hidden_320",
        "feature_shape":list(features.shape),
        "feature_dtype":"torch.float16",
        "feature_sha256":sha_file(tensor_file),
        "feature_file_bytes":tensor_file.stat().st_size,
        "feature_reload_bitwise_equal":True,
        "captured_unique_prompt_count":len(feature.features),
        "captured_train32_count":len(ordered),
        "native_CPU_readonly_last_layer_forwards":feature.forward_calls,
        "NRA2_gate":0.0,
        "existing_456_full_EOS":456,
        "additional_128_full_EOS":128,
        "new32_first":16,
        "new32_full_EOS":0,
        "protected_584_digest_matches_S194_and_S2":True,
        "new32_digest_matches_S194_and_S2":True,
        "NRA2_hidden_to_graph_adapter_trained":False,
        "NRA2_output_influences_generation":False,
        "candidate_CPU_speed_tested":False,
        "candidate_all_in_bpp_tested":False,
        "Jessica_LIVE_Render_Supabase_APK_untouched":True,
        "candidate_model_accepted":False,
    }
    (out/"NRA2_S3A_REAL_CPU_32_TRAIN_FEATURE_AUDIT.json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in metadata.items()
                      if k!="dataset_question_sha256_sorted"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
