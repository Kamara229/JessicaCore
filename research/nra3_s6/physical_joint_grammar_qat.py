"""NRA2 3.0 S6: real frozen 43M CPU hidden -> unified physically-paid CB59 QAT.

The ONLY eligible observations are the twelve SHA-locked, hand-annotated
Russian UD TRAIN prompts from S4, with genuine read-only post-GT layer8
hidden320 from physically verified S5. NO EVAL, QA, Jessica chat, LIVE,
or parent/GT backprop. The 43 CB59 5-bit index arrays stay immutable;
optimizer changes only existing physical FP16 centroid and row-scale fields.

Temporary dense differentiable surrogate exists ONLY during TRAIN; it is
never saved. Post-train inference is only physical CB59 C CPU, same Green
Tunnel compatibility contract, and output gate = 0. No generalization or
616-J7 candidate improvement is claimed by this TRAIN-fit experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from safetensors.torch import load_file

from direct_codebook_v59 import CB59
from v176.codec import replace_fields
from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s1.physical_cb59 import reconstruct_state
from research.nra3_s3.paid_codebook_qat import CB59PaidCodebook
from research.nra3_s4.russian_ud_train_only import (
    TRAIN12_CANONICAL_SHA256,RussianTrainSentence,build_train,commitment,
)
from research.nra3_s4.complete_prefix_training_contract import (
    CausalAlignment,AlignedWord,train_only_multitask_loss,
)


S5_ORIGINAL_RUN=43699866531
S5_SOURCE_SHA256="ba04d54d399419ec857d01b8401af868213a7f8b2c219d384986b8cd08b1ba1f"
S194_FP16_SHA256="b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2"


def _ok(condition,why):
    if not condition:
        raise ValueError("NRA3 S6 real TRAIN source: "+why)


@dataclass(frozen=True)
class OriginalRussianTrainSample:
    row:RussianTrainSentence
    hidden320:torch.Tensor
    aligned:CausalAlignment


def load_pinned_real43m_train12(feature_path:Path,report_path:Path,
                               *,expected_43m_feature_sha256=S5_SOURCE_SHA256):
    """No generated answer, gold label or EVAL read from original S5 artifact."""
    feature_path,report_path=Path(feature_path),Path(report_path)
    _ok(feature_path.is_file() and report_path.is_file(),
        "S5 immutable physical files required")
    raw=feature_path.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    _ok(sha==expected_43m_feature_sha256 and sha==S5_SOURCE_SHA256,
        "S5 genuine 43M CPU feature SHA not pinned")
    evidence=json.loads(report_path.read_text(encoding="utf-8"))
    expected={
        "stage":"NRA3_S5_REAL_ORIGINAL_43M_CPU_GT_POST_LAYER8_RUSSIAN_TRAIN12_SHADOW_ONLY",
        "S4_RU_UD_TRAIN12_CANONICAL_SHA256":TRAIN12_CANONICAL_SHA256,
        "S194_ORIGINAL_CHECKPOINT_SHA256":S194_FP16_SHA256,
        "actual_fp16_feature_sha256":sha,
        "actual_genuine_43M_CPU_prefills":12,
        "observed_layer_index_zero_based":7,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "NRA3_output_gate":0.0,
        "NRA3_candidate_accepted":False,
        "all_616_native_CPU_generated_token_digests_unchanged":True,
        "GT_residuals_before_readport_in_same_active_S93_scope":True,
        "real_tokenizer_chat_offsets_verified":True,
        "all_gold_labels_excluded_from_frozen_decoder":True,
        "actual_NRA3_6M_weights_trained":False,
        "five_consumed_independent_EVAL128_not_loaded":True,
        "sixth_reserved_S3CI_EVAL128_not_loaded":True,
    }
    _ok(all(evidence.get(k)==v for k,v in expected.items()),
        "S5 physically verified frozen source, native J7, or zero gate mismatch")
    tensors=load_file(str(feature_path),device="cpu")
    _ok(set(tensors)=={"token_hidden320","input_ids","token_mask","token_lengths"},
        "S5 extra hidden or answer tensor forbidden")
    hidden=tensors["token_hidden320"]
    ids=tensors["input_ids"]
    mask=tensors["token_mask"]
    lengths=tensors["token_lengths"]
    _ok(hidden.ndim==3 and hidden.shape[0]==12 and hidden.shape[2]==320 and
        hidden.dtype==torch.float16 and ids.shape==mask.shape==hidden.shape[:2] and
        ids.dtype in (torch.int32,torch.int64) and mask.dtype==torch.bool and
        lengths.shape==(12,) and not bool((~mask[:,0]).any()) and
        bool(torch.isfinite(hidden).all()) and
        evidence["actual_hidden320_shape"]==list(hidden.shape) and
        evidence["true_train12_token_counts"]==lengths.tolist(),
        "true 43M CPU hidden token ABI mismatch")
    rows=build_train()
    _ok(commitment(rows)["canonical_training_rows_sha256"]==
        TRAIN12_CANONICAL_SHA256,"S4 immutable UD labels modified")
    ids_list=[]
    out=[]
    maps=evidence["wordpiece_alignment"]
    _ok(isinstance(maps,list) and len(maps)==12,"missing true tokenizer word alignments")
    for i,row in enumerate(rows):
        n=int(lengths[i])
        _ok(1<n<=hidden.shape[1] and torch.equal(
            mask[i],torch.arange(hidden.shape[1])<n) and
            bool((ids[i,n:]==-1).all()) and
            bool((hidden[i,n:]==0).all()),
            "right pad, token IDs or original CPU feature length drift")
        token_ids=tuple(int(x) for x in ids[i,:n].tolist())
        _ok(all(x>=0 for x in token_ids),
            "real tokenizer prefill token IDs invalid")
        ids_list.append(list(token_ids))
        m=maps[i]
        _ok(m.get("text_sha256")==row.sha256() and
            m.get("token_count")==n and
            len(m["word_last_token_positions"])==len(row.words) and
            len(m["word_token_positions"])==len(row.words),
            "S5 wordpiece map doesn't correspond to original UD TRAIN")
        word_map=[]
        used=set()
        for j,(last,positions) in enumerate(zip(
                m["word_last_token_positions"],m["word_token_positions"])):
            _ok(type(last) is int and type(positions) is list and positions
                and all(type(x) is int and 0<=x<n for x in positions)
                and last==max(positions) and
                not any(x in used for x in positions),
                "S5 original real-tokenizer word offsets overlapping or future")
            used.update(positions)
            word_map.append(AlignedWord(j+1,tuple(positions),last))
        _ok(all(word_map[j].last_token<word_map[j+1].last_token
                for j in range(len(word_map)-1)),
            "S5 real tokenizer UD word order changed")
        align=CausalAlignment(row.sha256(),n,token_ids,tuple(word_map),n-1)
        out.append(OriginalRussianTrainSample(
            row,hidden[i,:n].float().contiguous(),align))
    canon=json.dumps(ids_list,sort_keys=True,separators=(",",":")).encode()
    _ok(hashlib.sha256(canon).hexdigest()==
        evidence["actual_token_ids_lengths_sha256"],
        "S5 native tokenizer physical SHA list drift")
    return tuple(out),evidence


class RussianJointSupervision(nn.Module):
    """One unified NRA3, no external adapter, no additional neural heads."""
    def __init__(self,config:NRA3Config):
        super().__init__()
        self.area=NRA3NeuralArea(config)

    def forward(self,hidden320,row,alignment):
        return train_only_multitask_loss(self.area,hidden320,row,alignment)


class PhysicallyPaidNRA3QAT(nn.Module):
    """CB59 source-index locked differentiable QAT of 43 existing matrices.

    Dense reference/Adam may exist transiently in this *TRAIN-only* class.
    Never save torch state_dict or export an FP32 checkpoint. Only the
    original Calibri codec physically serializes all trained paid fields.
    """
    def __init__(self,physical_cb59:bytes,config:NRA3Config):
        super().__init__()
        self.config=config
        self.original=physical_cb59
        self.source=CB59(physical_cb59)
        self._host=RussianJointSupervision(config).cpu().eval()
        full=reconstruct_state(physical_cb59,self._host.area)
        self._host.area.load_state_dict(full,strict=True)
        self._host.requires_grad_(False)
        keys=tuple(sorted(self.source.handles))
        # Freeze every full model weight. The ONLY trainable parameters
        # are already paid FP16 books and row scales, via float32 optimizers.
        self.names=keys
        self.books=nn.ModuleList([
            CB59PaidCodebook(self.source,k) for k in keys])
        self.output_gate=0.
        self.candidate_accepted=False
        self._source_names=set(self._host.area.state_dict())
        _ok(self._source_names==set(self.source.meta["tensor_descriptors"])
            and len(self.names)==43 and all(x.endswith(".weight")
                  or x.endswith("in_proj_weight") for x in self.names),
            "single original NRA3 6M 43-matrix source identity invalid")
        _ok(sum(p.numel() for p in self._host.area.parameters())==6_106_154
            or config!=NRA3Config(),"full NRA3 initial physical capacity drift")
        # Static 1D weights are not trainable; they remain in the exact
        # initial CB59 physical archive unchanged.
        _ok(all(not p.requires_grad for p in self._host.parameters()),
            "unexpected parent or dense area optimizer target")

    def forward(self,hidden320,row,aligned):
        if self.output_gate!=0. or self.candidate_accepted:
            raise RuntimeError("S6 packed grammar QAT is strictly zero-gate")
        if (hidden320.ndim!=2 or hidden320.shape[1]!=320
                or hidden320.dtype!=torch.float32 or
                hidden320.device.type!="cpu" or
                not bool(torch.isfinite(hidden320).all())):
            raise ValueError("S6 true 43M token hidden requires CPU [T,320] float32")
        patch={"area."+name:book.training_weight()
               for name,book in zip(self.names,self.books)}
        return functional_call(self._host,patch,
                               (hidden320.unsqueeze(0),row,aligned))

    def save_full_paid_physical_cb59(self)->bytes:
        if self.output_gate!=0. or self.candidate_accepted:
            raise RuntimeError("S6 cannot export active model")
        values={}
        lookup=dict(zip(self.names,self.books))
        for name,desc in self.source.meta["tensor_descriptors"].items():
            if desc["dtype"]=="CB59_K32V8":
                centroids,scales=lookup[name].save_paid_fields()
                values[name]={"centroids":centroids,"row_scales":scales}
            elif desc["dtype"]=="f4":
                values[name]={"raw":self.source.raw[name]}
            else:
                raise ValueError("S6 unsupported source matrix encoding")
        paid=replace_fields(self.source,values)
        _ok(len(paid)==len(self.original),"new physical source grew in bytes")
        after=CB59(paid)
        _ok(after.meta==self.source.meta and
            after.meta["original_unique_parameter_count"]==6_106_154,
            "repacked source metadata/count drift")
        changed=set()
        for key in self.names:
            left,right=self.source.handles[key],after.handles[key]
            _ok(left.packed.tobytes()==right.packed.tobytes(),
                "5bit discrete source indices changed in physical archive")
            if (not np.array_equal(left.centroids,right.centroids)
                    or not np.array_equal(left.row_scales,right.row_scales)):
                changed.add(key)
        for k in self.source.raw:
            _ok(np.array_equal(self.source.raw[k],after.raw[k]),
                "original source 1D weight changed")
        _ok(any(k.startswith("grammar.") for k in changed) and
            any(k.startswith("semantic.") for k in changed) and
            any(k.startswith("upos_head") for k in changed) and
            any(k.startswith("role_head") for k in changed),
            "real physical grammar+semantics source did not train jointly")
        return paid

    def physical_change_counts(self,after:bytes)->dict:
        packed=CB59(after)
        modified=[name for name in self.names if
                  not np.array_equal(self.source.handles[name].centroids,
                                     packed.handles[name].centroids)
                  or not np.array_equal(self.source.handles[name].row_scales,
                                        packed.handles[name].row_scales)]
        return {
            "changed_physically_paid_2d_source_operators":len(modified),
            "grammar_matrix_changes":sum(k.startswith("grammar.") for k in modified),
            "semantic_matrix_changes":sum(k.startswith("semantic.") for k in modified),
            "upos_head_changed":"upos_head.weight" in modified,
            "role_head_changed":"role_head.weight" in modified,
            "original_fixed_5bit_matrix_index_sets":len(self.names),
            "original_all_1d_source_bytes_unchanged":True,
        }


def numerical_trainonly_losses(qat:PhysicallyPaidNRA3QAT,
                               rows:tuple[OriginalRussianTrainSample,...]):
    qat.eval()
    with torch.no_grad():
        arr=[float(qat(row.hidden320,row.row,row.aligned).total)
             for row in rows]
    if len(arr)!=12 or any(not math.isfinite(x) for x in arr):
        raise AssertionError("S6 authentic source loss undefined")
    return sum(arr)/len(arr)


def train_locked_12_real_hidden(qat:PhysicallyPaidNRA3QAT,
                                samples:tuple[OriginalRussianTrainSample,...],
                                *,steps=24,lr=.012):
    """TRAIN fit only, no independent heldout, no QA, no existing parent J7."""
    _ok(len(samples)==12 and 1<=steps<=36 and 0<lr<=.04,
        "only bounded frozen original RU TRAIN12 learning permitted")
    _ok(tuple(x.row.sha256() for x in samples)==
        tuple(x.sha256() for x in build_train()),
        "training source identity or order drift")
    torch.manual_seed(306006)
    torch.set_num_threads(2)
    opt=torch.optim.Adam(qat.books.parameters(),lr=lr)
    before=numerical_trainonly_losses(qat,samples)
    trace=[]
    min_loss=before
    best=None
    for step in range(steps):
        item=samples[step%12]
        # Fail-closed per-step original immutable TRAIN membership.
        _ok(item.row.split=="NRA3_S4_RU_UD_TRAIN_ONLY" and
            item.row.sha256() in {r.sha256() for r in build_train()},
            "unknown input entered optimizer step")
        opt.zero_grad(set_to_none=True)
        losses=qat(item.hidden320,item.row,item.aligned)
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError("S6 nonfinite actual Russian TRAIN loss")
        losses.total.backward()
        updates=[
            b.centroids.grad for k,b in zip(qat.names,qat.books)
            if (k.startswith("grammar.") or k.startswith("semantic."))
            and b.centroids.grad is not None]
        _ok(updates and any(bool((g!=0).any()) for g in updates),
            "no real joint grammar/semantics CB59 gradients")
        torch.nn.utils.clip_grad_norm_(qat.books.parameters(),.5)
        opt.step()
        if (step+1)%12==0 or step==steps-1:
            current=numerical_trainonly_losses(qat,samples)
            trace.append({"steps":step+1,"train12_mean_multitask_loss":current})
            if current<min_loss:
                min_loss=current
                best={k:v.detach().clone() for k,v in qat.books.state_dict().items()}
    _ok(best is not None and min_loss<before and
        all(math.isfinite(x["train12_mean_multitask_loss"]) for x in trace),
        "S6 authentic TRAIN12 no measured improvement (do not save candidate)")
    qat.books.load_state_dict(best,strict=True)
    after=numerical_trainonly_losses(qat,samples)
    _ok(after<=min_loss+1e-5 and after<before,
        "S6 best physical TRAIN state failed to reload")
    return {"before_train12_multitask_loss":before,
            "after_train12_multitask_loss":after,
            "relative_training_only_loss_reduction":(before-after)/before,
            "optimizer_steps":steps,
            "optimizer_trace":trace,
            "original_parent_43M_weights_updated":False,
            "grammar_semantics_train_only":True,
            "rational_generalization_evidence":False,
            "output_gate":0,
            "candidate_accepted":False}
