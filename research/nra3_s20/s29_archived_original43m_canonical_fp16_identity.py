"""S29: verify canonical paid tensor identity for actual S22d/S26 archived FP16.

S28 physical genuine 43M+GT CPU training proved exact same FP32 and
canonical FP16 weights at 0/1/5/40/240 steps despite varying model payload
SHA from the nested legacy safetensors serializer. This research-only stage
keeps BOTH: the original raw physical artifact SHA for integrity, and a
new sorted named-parameter canonical FP16 fingerprint for semantic identity.
It does not rewrite old 616 J7, S19d, the model architecture or LIVE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph,full_parameter_count,semantic_parameter_count,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,
)
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)

FROZEN_ARCHIVES={
    "S22d":{
        "sha":"bd56293aa6ada2411152f0ac528633315731c39de6e74a759704577defbadcbb",
        "full_paid_params":12381,"head_paid_params":3049,
        "cls":PaidSemanticEventGraph,
    },
    "S26":{
        "sha":"4819c8b0dfc402bddf036f16e04f85486c51a9a6600af63374cf40a1f6c00951",
        "full_paid_params":12384,"head_paid_params":3052,
        "cls":EventGroundedInversionGraph,
    },
}
EXPECTED_PARENT_SHA="928f735a2018604db2ffe649799778df3805162b79a989a36e282dc3cb61c774"


def require(check,detail):
    if not check:
        raise ValueError("S29 archived paid original43M fp16 fingerprint: "+detail)


def sha(raw:bytes):
    return hashlib.sha256(raw).hexdigest()


def audit_actual_archived_model(stage:str,path:Path):
    require(stage in FROZEN_ARCHIVES and path.is_file(),"not a supported original archived model")
    spec=FROZEN_ARCHIVES[stage]
    stored=path.read_bytes()
    require(sha(stored)==spec["sha"],"original archived full paid physical file SHA mismatch")
    model=spec["cls"].from_physical(stored).eval()
    require(full_parameter_count(model)==spec["full_paid_params"]
            and semantic_parameter_count(model)==spec["head_paid_params"]
            and model.output_gate==0
            and not model.candidate_accepted and not model.activation_allowed
            and not model.parent.candidate_accepted and model.parent.output_gate==0
            and all(not p.requires_grad for p in model.parameters()),
            "original full paid FP16 parameters/source/inactive gates modified")
    canonical=canonical_paid_semantic_fp16_sha(model)
    stable_head=sha(model.head_physical())
    sha_archive=sha(stored)
    sampled=[]
    for _ in range(6):
        # Important: new raw bytes may differ even for the SAME model;
        # source weights must remain identical. Do not confuse integrity
        # artifact SHA with a cross-run semantic reproducibility digest.
        again=model.physical_bytes()
        decoded=spec["cls"].from_physical(again)
        require(canonical_paid_semantic_fp16_sha(decoded)==canonical
                and sha(decoded.head_physical())==stable_head
                and decoded.output_gate==0
                and not decoded.candidate_accepted
                and not decoded.activation_allowed,
                "full physical reserialization modified the actual numeric paid FP16 model")
        sampled.append(sha(again))
    return {
        "model":stage,
        "source_original_raw_physical_SHA256":sha_archive,
        "original_raw_sha_verified":True,
        "full_paid_FP16_parameter_count":spec["full_paid_params"],
        "semantic_trainable_paid_FP16_parameter_count":spec["head_paid_params"],
        "canonical_named_paid_FP16_numeric_SHA256":canonical,
        "head_paid_FP16_numeric_SHA256":stable_head,
        "same_original_model_reserialized_raw_SHA256_samples":sampled,
        "distinct_same_model_raw_serializer_SHA_count":len(set(sampled)),
        "same_model_numeric_canonical_digest_invariant_across_reserialization":True,
        "physical_archive_bytes_NOT_REPLACED":True,
        "no_candidate_or_decoder_activation":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s22d-model",type=Path,required=True)
    p.add_argument("--s26-model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"never overwrite source/archived model")
    baseline=audit_actual_archived_model("S22d",a.s22d_model)
    improved=audit_actual_archived_model("S26",a.s26_model)
    require(baseline["canonical_named_paid_FP16_numeric_SHA256"] !=
            improved["canonical_named_paid_FP16_numeric_SHA256"],
            "different architectures must not impersonate the same model")
    result={
        "stage":"NRA3_S29_ACTUAL_ARCHIVED_S22D_S26_CANONICAL_FP16_IDENTITY_AUDIT",
        "real_original43M_GT_S22c_capture_run":44191302703,
        "original_S22d_training_run":44192186916,
        "original_S26_training_run":44196720655,
        "S28_genuine_original43M_GT_determinism_proof_run":44200775382,
        "archive_integrity_vs_numerical_model_equivalence":"Raw physical SHA authenticates one saved artifact; sorted paid FP16 canonical SHA authenticates numerical weights irrespective of nondeterministic safetensors header layout.",
        "historical_40fold_weights_not_saved_and_not_claimed_reproducible":True,
        "sources":{"S22d":baseline,"S26":improved},
        "no_training_or_independent_EVAL":True,
        "LIVE_Render_Supabase_Jessica_APK_original_Calibri_untouched":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"NRA3_S29_ARCHIVED_S22D_S26_CANONICAL_PAID_FP16_PROVENANCE.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8")
    for name,r in result["sources"].items():
        print("S29_ARCHIVED_CANONICAL_FP16_DIGEST",name,
              r["canonical_named_paid_FP16_numeric_SHA256"],
              "RAW_SERIALIZER_UNIQUE",r["distinct_same_model_raw_serializer_SHA_count"],
              "ARCHIVE_SHA",r["source_original_raw_physical_SHA256"],flush=True)
    print("S29_ORIGINAL43M_PAID_FP16_NUMERICAL_MODEL_IDENTITIES_SHA_VALIDATED",flush=True)


if __name__=="__main__":
    main()
