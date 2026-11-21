"""S119h: mathematically structured latent representation + transformation.

User hypothesis: let an extensible unified Calibri CORE solve learnable latent
problems instead of requiring every task-specific mapping in the core.
R is BEFORE the first core layer; T is AFTER final core layer, BEFORE
language-model vocabulary head. A direct identity residual path prevents
accidental loss of lexical evidence and exactly preserves original behavior
at zero initialization.

IMPORTANT: any LLM already uses mathematical vector representations.
This module studies a MORE STRUCTURED, LEARNED set of 4 latent slots,
not a claim to "convert text into mathematics" in some unprecedented sense.
T is a generative output-space adapter, NOT an inverse bijection of R.
Nothing is hardcoded as a symbolic answer lookup or external solver.

True original S119g neural child can be loaded; the causal core in this
FIRST experiment is a small SYNTHETIC MOCK so actual pre-core / post-core
placement and gradient coupling can be tested safely. S119h does NOT
run frozen actual Calibri 43M, alter actual CB59/GT, prove understanding,
generative J7, speed, or bpp. Original live gates stay zero.

No Jessica, LIVE, Render, Supabase, main, production, S118 GOLD,
previous spent developer test, or tool execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import (
    UnifiedCalibriWithNPA1,
    CHECKPOINT as S119G_CHECKPOINT,REPORT as S119G_REPORT,
    STAGE as S119G_STAGE,PREFIX as S119G_PREFIX,
)

STAGE="CALIBRI_S119H_PRECORE_STRUCTURED_REP_POSTCORE_GENERATIVE_TRANSFORMATION_JOINT_MOCK_ONLY"
CHECKPOINT="CALIBRI_S119H_REAL_S119G_PLUS_REP_TRANSFORM_FP16_TOY_TRAINONLY_UNACCEPTED.safetensors"
REPORT="CALIBRI_S119H_R_T_TWO_AREAS_CAUSAL_JOINT_MOCK_REPORT_NOT_QUALITY.json"
CHANNELS=320
SLOTS=4
SLOT_DIM=16


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NeuralRepresentationArea(nn.Module):
    """Input->4 learned latent mathematical slots->same 320D core interface.

    Residual identity keeps source spelling/detail available; no vocabulary
    replacement, no lossy fixed 64D bottleneck imposed on original core.
    Token-local transform prevents using unseen future tokens.
    """
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(CHANNELS)
        self.project=nn.Linear(CHANNELS,SLOTS*SLOT_DIM,bias=False)
        self.router=nn.Linear(CHANNELS,SLOTS,bias=False)
        self.fuse=nn.Linear(SLOTS*SLOT_DIM,CHANNELS,bias=False)
        nn.init.zeros_(self.fuse.weight)
        self.output_gate=0
        self.candidate_accepted=False
    def forward(self,embedding):
        check(self.output_gate==0 and not self.candidate_accepted
              and embedding.ndim==3 and embedding.shape[-1]==CHANNELS,
              "R must remain pre-core [B,T,320] unaccepted")
        z=self.norm(embedding)
        encoded=F.gelu(self.project(z)).reshape(
            z.shape[0],z.shape[1],SLOTS,SLOT_DIM)
        routes=self.router(z).softmax(-1).unsqueeze(-1)
        # Identity guarantees no information dropped at initialization.
        return embedding+self.fuse((routes*encoded).flatten(-2))


class NeuralTransformationArea(nn.Module):
    """Core+NRA3+NPA1 -> generative hidden for EXISTING native LM head.

    T is NOT required to invert R (output may contain new inferences).
    Learned optional graph/plan contextual side-channel is zero at init.
    """
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(CHANNELS)
        self.decode=nn.Linear(CHANNELS,SLOTS*SLOT_DIM,bias=False)
        self.fuse=nn.Linear(SLOTS*SLOT_DIM,CHANNELS,bias=False)
        self.semantic_context=nn.Linear(32+6,CHANNELS,bias=False)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.semantic_context.weight)
        self.output_gate=0
        self.candidate_accepted=False
    def forward(self,core_hidden,graph,plan,valid):
        check(self.output_gate==0 and not self.candidate_accepted
              and core_hidden.ndim==3 and core_hidden.shape[-1]==CHANNELS
              and valid.shape==core_hidden.shape[:2]
              and valid.dtype==torch.bool,
              "T must be post-core pre-logits native h320")
        last=valid.long().sum(-1)-1
        check(bool(valid.any(-1).all())
              and bool((valid==(
                  torch.arange(valid.shape[1],device=valid.device)[None,:]
                    <=last[:,None])).all()),
              "causal transform requires contiguous prefix, no future reads")
        output=core_hidden+self.fuse(F.gelu(self.decode(
            self.norm(core_hidden))))
        rows=torch.arange(last.numel(),device=last.device)
        event=graph.event_state_logits[rows,last].softmax(-1)
        plan_state=plan.state_trace[:,-1]
        auxiliary=self.semantic_context(torch.cat((event,plan_state),-1))
        # Only current last-token position receives plan synthesis; never
        # leak later tokens into previous positions in causal generation.
        addition=torch.zeros_like(output)
        addition[rows,last]=auxiliary
        return output+addition


class UnifiedCalibriWithRepresentationAndTransformation(UnifiedCalibriWithNPA1):
    """Old S119g full 116,777 weights + both NEW areas, ONE model."""
    def __init__(self):
        super().__init__()
        self.representation_area=NeuralRepresentationArea()
        self.transformation_area=NeuralTransformationArea()

    def load_real_s119g(self,checkpoint:dict):
        new_names=("representation_area.","transformation_area.")
        old={k[len(S119G_PREFIX):]:v.float()
             for k,v in checkpoint.items()
             if k.startswith(S119G_PREFIX)}
        expected={k for k in self.state_dict()
                  if not k.startswith(new_names)}
        check(len(old)==len(checkpoint)
              and set(old)==expected
              and sum(v.numel() for v in checkpoint.values())==116_777
              and all(v.dtype==torch.float16 for v in checkpoint.values()),
              "FULL real S119g checkpoint, NO lost original learned tensors")
        status=self.load_state_dict(old,strict=False)
        check(not status.unexpected_keys
              and set(status.missing_keys)==(
                  set(self.state_dict())-expected),
              "only the TWO new neural areas may initialize")
        for p in self.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        check(sum(p.numel()
                  for p in self.nra3_neural_weights.parameters())==53_751
              and not any(p.requires_grad for p in
                          self.nra3_neural_weights.parameters()),
              "frozen original 53,751 in NRA3 and eight minis kept")

    def joint_forward_on_core(self,all12,valid):
        """Core outputs [B,12,T,320], no second core R insertion here."""
        graph,relational,planning=super().forward(all12,valid)
        # Transform true last core state AFTER core computation and prior to
        # final output projection, with graph & learned planner side context.
        transformed=self.transformation_area(
            self.core_growth(all12)[:,-1],graph,planning,valid)
        return transformed,graph,relational,planning

    def new_joint_trainables(self):
        return (super().new_joint_trainables()
                +list(self.representation_area.parameters())
                +list(self.transformation_area.parameters()))


class Causal12LayerMock(nn.Module):
    """Synthetic causal test stand, emphatically NOT original Calibri/GT."""
    def __init__(self,vocab=37):
        super().__init__()
        self.input_embeddings=nn.Embedding(vocab,CHANNELS)
        self.layers=nn.ModuleList([
            nn.Sequential(nn.Linear(CHANNELS,8,bias=False),nn.Tanh(),
                          nn.Linear(8,CHANNELS,bias=False))
            for _ in range(12)])
        self.output_embeddings=nn.Linear(CHANNELS,vocab,bias=False)

    def encode12(self,emb):
        traces=[]
        x=emb
        for layer in self.layers:
            x=x+0.05*layer(x)
            traces.append(x)
        return torch.stack(traces,1)

    def unmodified_logits(self,ids):
        base=self.encode12(self.input_embeddings(ids))
        return self.output_embeddings(base[:,-1])


def model_logits(mock:Causal12LayerMock,
                 unified:UnifiedCalibriWithRepresentationAndTransformation,
                 ids:torch.Tensor):
    """ACTUAL hook-order simulation: native ids -> embedding -> R -> core
    -> all12 for NRA3/NPA1 -> T -> old vocabulary head. No external router.
    """
    check(ids.ndim==2 and ids.shape[1]>=2
          and ids.dtype==torch.long and
          bool((ids>=0).all()) and bool((ids<mock.input_embeddings.num_embeddings).all()),
          "toy math task valid mock native IDs")
    embeddings=mock.input_embeddings(ids)
    represented=unified.representation_area(embeddings)
    all12=mock.encode12(represented)
    mask=torch.ones(ids.shape,dtype=torch.bool,device=ids.device)
    transformed,graph,relation,plan=unified.joint_forward_on_core(all12,mask)
    return mock.output_embeddings(transformed),graph,relation,plan


def synthetic_joint_training(mock,unified):
    """Two real updates on artificial TRAIN-only arithmetic labels.

    A purely numeric diagnostic of differentiable three-way interaction,
    not a claim that the model learned arithmetic or natural-language logic.
    """
    torch.manual_seed(119119)
    x=torch.randint(0,37,(3,6),dtype=torch.long)
    label=((x[:,0]+2*x[:,1]+x[:,2])%37).long()
    target=torch.zeros((3,6,6),dtype=torch.float32)
    # Synthetic plan auxiliary only to guarantee NPA receives gradient,
    # NOT a ground truth general-purpose action trace.
    target[:,:,-1]=1.0
    old={k:v.detach().half().clone()
         for k,v in unified.nra3_neural_weights.state_dict().items()}
    with torch.inference_mode():
        baseline=mock.unmodified_logits(x)
        prior,g,_,p=model_logits(mock,unified,x)
        check(torch.equal(prior,baseline)
              and torch.equal(prior.argmax(-1),baseline.argmax(-1)),
              "both latent areas must have EXACT zero-initialization identity")
    opt=torch.optim.Adam(
        [*mock.parameters(),*unified.new_joint_trainables()],
        lr=0.0003)
    traces=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        logits,graph,relation,plan=model_logits(mock,unified,x)
        # Joint next symbol + latent planning + generic relational target
        loss=F.cross_entropy(logits[:,-1],label)
        loss=loss+0.2*F.mse_loss(plan.next_event_logits,torch.zeros_like(
                                 plan.next_event_logits))
        loss=loss+0.01*relation.square().mean()
        check(bool(torch.isfinite(loss)),
              "single original core R T NRA NPA objective finite")
        loss.backward()
        grads={
            "representation":unified.representation_area.fuse.weight.grad,
            "transformation":unified.transformation_area.fuse.weight.grad,
            "transformation_NPA_context":
                unified.transformation_area.semantic_context.weight.grad,
            "core_layer":mock.layers[0][0].weight.grad,
            "core_growth":unified.core_growth.up.grad,
            "graph_gate":unified.reasoning_role_gate.grad,
            "planner":unified.planning_action_area.pred_next_event.weight.grad,
        }
        for name,grad in grads.items():
            check(grad is not None and bool(torch.isfinite(grad).all())
                  and float(grad.abs().sum())>1e-12,
                  "one shared loss must reach "+name)
        traces.append({"step":step+1,"joint_synthetic_math_loss":float(loss.detach()),
                       "grad_l1":{name:float(g.abs().sum())
                                  for name,g in grads.items()}})
        opt.step()
    check(all(torch.equal(t.detach().half(),old[k])
              for k,t in unified.nra3_neural_weights.state_dict().items()),
          "ALL old 53751 original NRA weights cannot be changed")
    return traces


def run(prior:Path,prior_report:Path,out:Path):
    check(not out.exists() and prior.name==S119G_CHECKPOINT
          and prior_report.name==S119G_REPORT,
          "only actual successful S119g unified model source")
    witness=json.loads(prior_report.read_text(encoding="utf8"))
    check(witness["stage"]==S119G_STAGE
          and witness["physical_single_child_SHA256"]==sha(prior)
          and witness["physical_single_child_checkpoint_bytes"]==prior.stat().st_size
          and witness["physical_single_child_all_FP16_parameters"]==116_777
          and witness["all_53751_original_NRA2_NRA3_params_preserved_byte_exact"]
          and witness["candidate_accepted"] is False
          and witness["output_gate"]==0,
          "source pin requires S119g real trained provenance report")
    torch.manual_seed(119)
    torch.set_num_threads(2)
    orig=load_file(str(prior),device="cpu")
    model=UnifiedCalibriWithRepresentationAndTransformation()
    model.load_real_s119g(orig)
    mock=Causal12LayerMock()
    gradients=synthetic_joint_training(mock,model)
    physical={S119G_PREFIX+k:v.detach().half().cpu().contiguous()
              for k,v in model.state_dict().items()}
    check(set(orig).issubset(physical)
          and all(torch.equal(physical[k],v) for k,v in orig.items()
                  if ".nra3_neural_weights." in k),
          "original trained S119g NRA3 FP16 weights untouched")
    out.mkdir(parents=True)
    saved=out/CHECKPOINT
    save_file(physical,str(saved))
    roundtrip=load_file(str(saved),device="cpu")
    check(set(physical)==set(roundtrip)
          and all(torch.equal(roundtrip[k],v) for k,v in physical.items()),
          "full old+R+T neural child checkpoint physically roundtrips")
    return_report={
        "stage":STAGE,
        "S119g_parent_full_FP16_SHA256":sha(prior),
        "S119g_inherited_original_53751_FP16_unchanged":True,
        "new_representation_area_neural_params":
            sum(p.numel() for p in model.representation_area.parameters()),
        "new_transformation_area_neural_params":
            sum(p.numel() for p in model.transformation_area.parameters()),
        "new_neural_areas_are_in_ONE_model":True,
        "representation_applied_before_mock_CORE":True,
        "transformation_applied_after_mock_CORE_before_lm_head":True,
        "transformation_is_NOT_mathematical_inverse_of_representation":True,
        "four_latent_math_slots_per_token":SLOTS,
        "residual_identity_preserves_full_h320_input":True,
        "synthetic_math_only_joint_train_trace":gradients,
        "old_model_output_identical_before_training_on_mock":True,
        "real_original_calibri43m_cb59_gt_forward_NOT_USED":True,
        "original_616_generated_tokens_not_retested":True,
        "independent_natural_language_quality_NOT_PROVEN":True,
        "whole_model_CPU_SPEED_NOT_MEASURED":True,
        "complete_physical_bpp_NOT_MEASURED":True,
        "protected_S118_SYNTAGRUS_B_GOLD_UNOPENED":True,
        "unified_child_FP16_num_parameters":
            sum(t.numel() for t in physical.values()),
        "unified_child_FP16_file_bytes":saved.stat().st_size,
        "unified_child_FP16_sha256":sha(saved),
        "candidate_accepted":False,"activation_allowed":False,
        "output_gate":0,
        "decision":"RESEARCH_ONLY_R_CORE_T_CONTRACT_AND_JOINT_GRADIENT_NOT_UNIVERSAL_QUALITY",
    }
    (out/REPORT).write_text(json.dumps(return_report,sort_keys=True,
                     indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("S119H_R_CORE_T_ARCHITECTURE_JOINT_GRADIENT_MOCK "+
          json.dumps({
            "old_S119g_params":sum(v.numel() for v in orig.values()),
            "representation_new":return_report["new_representation_area_neural_params"],
            "transformation_new":return_report["new_transformation_area_neural_params"],
            "all_params":return_report["unified_child_FP16_num_parameters"],
            "file_bytes":return_report["unified_child_FP16_file_bytes"],
            "all_components_joint_grads":True,
            "genuine_original_decoder_integrated":False,
            "candidate_accepted":False},sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--prior-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.prior,args.prior_report,args.out)

if __name__=="__main__":
    main()
