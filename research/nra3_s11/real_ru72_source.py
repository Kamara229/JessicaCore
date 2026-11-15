"""NRA3 S11: locked TRUE original 43M S10 hidden for 72 new Russian TRAIN.

Gold annotations originate exclusively from S9's SHA-pinned TRAIN source.
Five spent EVAL128 and reserved sixth S3CI EVAL128 never read. Verify
physical safetensors bytes, original tokenizer IDs, wordpiece offsets,
protected J7 unchanged and zero gate before forming any router TRAIN labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s9.new_ru_train72 import (
    NewRussianTrain,S9_TRAIN72_CANONICAL_SHA256,
    build_s9_train72,seal_train72,
)

S10_CAPTURE_RUN=43732621116
S10_FEATURE_SHA="ed262984c0fd48b60987308e7b135be7d05d3e5434fe9923f373aec178ff2b46"
S194_ORIGINAL_FP16_SHA="b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2"
S10_TOKEN_ID_SHA="94296e293a1f7632b65c697c520371719314da5566c63532f0ba871f40f5f3b6"


@dataclass(frozen=True)
class RealRussian72:
    row:NewRussianTrain
    hidden320:torch.Tensor
    aligned:CausalAlignment


def checked_real_train72(hidden_path:Path,report_path:Path):
    hidden_path=Path(hidden_path)
    report_path=Path(report_path)
    if not hidden_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("S11 physically captured S10 original 43M missing")
    digest=hashlib.sha256(hidden_path.read_bytes()).hexdigest()
    if digest!=S10_FEATURE_SHA:
        raise ValueError("S11 original 72 CPU hidden SHA differs from S10")
    report=json.loads(report_path.read_text(encoding="utf-8"))
    expected={
        "stage":"NRA3_S10_TRUE_FROZEN_CALIBRI_43M_POST_GT_LAYER8_NEW_RUSSIAN_TRAIN72_SHADOW_ONLY",
        "S9_RUSSIAN_UD_TRAIN72_CANONICAL_SHA256":S9_TRAIN72_CANONICAL_SHA256,
        "S194_ORIGINAL_CHECKPOINT_SHA256":S194_ORIGINAL_FP16_SHA,
        "actual_fp16_feature_sha256":S10_FEATURE_SHA,
        "actual_token_ids_lengths_sha256":S10_TOKEN_ID_SHA,
        "real_source_train72_count":72,
        "actual_genuine_43M_CPU_prefills":72,
        "observed_layer_index_zero_based":7,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "all_616_native_CPU_generated_token_digests_unchanged":True,
        "GT_residuals_before_readport_in_same_active_S93_scope":True,
        "all_gold_labels_excluded_from_frozen_decoder":True,
        "real_tokenizer_chat_offsets_verified":True,
        "true_real_Calibri_token_alignment_all_72":True,
        "actual_NRA3_6M_weights_trained":False,
        "NRA3_output_gate":0.0,
        "NRA3_candidate_accepted":False,
        "five_consumed_independent_EVAL128_not_loaded":True,
        "sixth_reserved_S3CI_EVAL128_not_loaded":True,
        "no_Render_Supabase_APK_LIVE_or_Jessica_chat":True,
    }
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError("S11 frozen original 43M J7/GT or TRAIN72 report drift")
    tensor=load_file(str(hidden_path),device="cpu")
    if set(tensor)!={"token_hidden320","input_ids","token_mask","token_lengths"}:
        raise ValueError("S11 cannot read EVAL/gold/other tensor from S10 physical")
    h=tensor["token_hidden320"];ids=tensor["input_ids"]
    mask=tensor["token_mask"];lengths=tensor["token_lengths"]
    if (h.ndim!=3 or h.shape[0]!=72 or h.shape[-1]!=320 or h.dtype!=torch.float16
        or h.shape[:2]!=ids.shape or ids.shape!=mask.shape
        or lengths.shape!=(72,) or ids.dtype!=torch.int32 or
        mask.dtype!=torch.bool or
        not bool(torch.isfinite(h).all()) or
        h.shape!=tuple(report["actual_hidden320_shape"]) or
        lengths.tolist()!=report["true_train72_token_counts"]):
        raise ValueError("S11 original 72-by-token hidden shape/dtype/padding mismatch")
    rows=build_s9_train72()
    if seal_train72()["canonical_train72_sha256"]!=S9_TRAIN72_CANONICAL_SHA256:
        raise ValueError("S11 user TRAIN corpus SHA changed")
    result=[]
    ident=[]
    wordmap=report.get("wordpiece_alignment")
    if not isinstance(wordmap,list) or len(wordmap)!=72:
        raise ValueError("S11 cannot map actual Calibri wordpiece offsets to 72 TRAIN")
    for i,row in enumerate(rows):
        n=int(lengths[i])
        if (not 1<n<=h.shape[1] or
            not bool(torch.equal(mask[i],torch.arange(h.shape[1])<n)) or
            not bool((ids[i,n:]==-1).all()) or
            not bool((h[i,n:]==0).all()) or
            not bool((ids[i,:n]>=0).all())):
            raise ValueError("S11 native CPU hidden has missing source or nonzero pad")
        tokens=tuple(int(k) for k in ids[i,:n].tolist())
        ident.append(list(tokens))
        mapping=wordmap[i]
        if (mapping.get("text_sha256")!=row.sha256() or
            mapping.get("token_count")!=n or
            len(mapping.get("word_last_token_positions",[]))!=len(row.words) or
            len(mapping.get("word_token_positions",[]))!=len(row.words)):
            raise ValueError("S11 original 72 annotation/prompt physical order drift")
        aligned_words=[]
        used=set()
        for j,(last,positions) in enumerate(zip(
            mapping["word_last_token_positions"],
            mapping["word_token_positions"])):
            if (type(last) is not int or not isinstance(positions,list) or
                not positions or any(type(k) is not int or k<0 or k>=n
                                     or k in used for k in positions) or
                last!=max(positions)):
                raise ValueError("S11 overlapping tokenization or future UD word")
            used.update(positions)
            aligned_words.append(AlignedWord(j+1,tuple(positions),last))
        if any(aligned_words[k].last_token>=aligned_words[k+1].last_token
               for k in range(len(aligned_words)-1)):
            raise ValueError("S11 out-of-order actual wordpiece positions")
        align=CausalAlignment(row.sha256(),n,tokens,tuple(aligned_words),n-1,False)
        result.append(RealRussian72(row,h[i,:n].float().contiguous(),align))
    canonical=json.dumps(ident,sort_keys=True,separators=(",",":")).encode()
    if hashlib.sha256(canonical).hexdigest()!=S10_TOKEN_ID_SHA:
        raise ValueError("S11 frozen original CPU real token sequences SHA drift")
    return tuple(result),report
