"""S119f: ONE in-Calibri child jointly learns core growth and NRA3-R reasoning.

Original twelve-layer CB59 + Green Tunnel are represented by SHA-locked
S114 actual CPU hidden captures. This is a first neural gradient integration
proof on genuine source features, NOT a true original decoder forward or a
candidate CPU/J7/understanding certificate. The complete old 53,751 S115
neural child is preserved. All training uses old licensed TRAIN-A source ONLY,
without original/third-party GOLD. Research only: gate0, no activation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119b_S194_parent_S115_child_physical_budget import S115_EXPECTED_SHA256, audit
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import PARENT_H_SHA, PARENT_H, PARENT_REPORT, two_hop_parameter_free_event_relation_graph
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import NRA3BoundedStructuralTrustInCalibriWeights

STAGE="NRA3_S119F_ONE_CALIBRI_CORE12_GT_NRA3R_JOINT_SHADOW_NO_GOLD"
CHECKPOINT="NRA3_S119F_UNIFIED_CORE_AND_NRA3R_84479_FP16_RESEARCH_SHADOW.safetensors"
REPORT="NRA3_S119F_REAL12_TRAIN_A_JOINT_GRADIENT_ABI_REPORT.json"
PREFIX="model.nra3_neural_weights."

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SharedCore12(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Parameter(torch.randn(12,320,4)*0.02)
        self.up=nn.Parameter(torch.zeros(12,4,320))
    def forward(self,h):
        check(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==320,
              "actual original12 H320 required")
        return h+torch.einsum("bltr,lrh->blth",
            F.gelu(torch.einsum("blth,lhr->bltr",h,self.down)),self.up)

class UnifiedCalibri(nn.Module):
    """One in-weight research child, not separate independently trained models."""
    def __init__(self):
        super().__init__()
        self.core_growth=SharedCore12()
        self.nra3_neural_weights=NRA3BoundedStructuralTrustInCalibriWeights()
        # Opt-in confined to THIS research-only child: original frozen CB59
        # source itself remains detached, only NEW core residual has grad.
        self.nra3_neural_weights._unified_core_grad_only_research=True
        self.reasoning_role_gate=nn.Parameter(torch.zeros(8))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
    def load_original_s115(self,phys):
        old={k[len(PREFIX):]:v.float() for k,v in phys.items()
             if k.startswith(PREFIX)}
        check(len(old)==len(phys) and sum(v.numel() for v in old.values())==53751,
              "all S115 old learnt weights must remain present")
        self.nra3_neural_weights.load_state_dict(old,strict=True)
        for p in self.nra3_neural_weights.parameters():
            p.requires_grad_(False)
    def forward(self,h,valid):
        check(self.output_gate==0 and not self.candidate_accepted
              and not self.activation_allowed and
              self.nra3_neural_weights.output_gate==0,
              "UNIFIED research child MUST NOT touch original decoder")
        g=self.nra3_neural_weights(self.core_growth(h),valid)
        probabilities=g.event_state_logits.softmax(-1)
        a=g.role_arc_logits.softmax(-1)
        relation=torch.einsum("brts,bsc->brtc",a,probabilities)
        delta=relation-probabilities[:,None]
        r=g.event_state_logits+torch.einsum("r,brtc->btc",
              self.reasoning_role_gate.tanh(),delta)
        return g,r
    def new_joint_trainables(self):
        return [self.core_growth.down,self.core_growth.up,self.reasoning_role_gate]
    def attach_to_research_only(self,model):
        # Attaching this child is NOT a forward-hook or an activation.
        backbone=getattr(model,"model",None)
        layers=getattr(backbone,"layers",None)
        check(isinstance(backbone,nn.Module) and layers is not None
              and len(layers)==12 and
              not hasattr(backbone,"nra3_calibri_joint"),
              "only an isolated research Calibri12 can register joint child")
        backbone.add_module("nra3_calibri_joint",self)

def train_joint(c,h,valid):
    c.eval()
    old=c.nra3_neural_weights
    before={k:v.detach().half().clone() for k,v in old.state_dict().items()}
    with torch.inference_mode():
        baseline=old(h,valid)
        graph,pred=c(h,valid)
        check(torch.equal(graph.role_arc_logits,baseline.role_arc_logits)
              and torch.equal(graph.mini_area_logits,baseline.mini_area_logits)
              and torch.equal(pred,baseline.event_state_logits),
              "initial rank4 zero UP and graph gate must equal ALL old S115 outputs")
        target=two_hop_parameter_free_event_relation_graph(baseline,valid)
        target=(target/target.sum(-1,keepdim=True).clamp_min(1.e-9)).detach()
    opt=torch.optim.Adam(c.new_joint_trainables(),lr=.001)
    losses=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        g,p=c(h,valid)
        l=-(target*F.log_softmax(p,-1)).sum(-1)[valid].mean()
        l=l+0.01*(g.mini_area_logits[valid]-
                   baseline.mini_area_logits.detach()[valid]).square().mean()
        check(bool(torch.isfinite(l)),"nonfinite unified source-only proxy loss")
        l.backward()
        u=c.core_growth.up.grad
        r=c.reasoning_role_gate.grad
        check(u is not None and r is not None and
              float(u.abs().sum())>1.e-9 and float(r.abs().sum())>1.e-9,
              "both core and in-model graph must receive SAME objective gradient")
        losses.append({"step":step+1,"proxy_loss":float(l.detach()),
                       "core_up_gradient_l1":float(u.abs().sum()),
                       "graph_gate_gradient_l1":float(r.abs().sum())})
        opt.step()
    check(bool(torch.count_nonzero(c.core_growth.up)) and
          bool(torch.count_nonzero(c.reasoning_role_gate)) and
          bool(torch.count_nonzero(c.core_growth.down.grad)),
          "both new core and new graph actually trained")
    check(all(torch.equal(v.detach().half(),before[k])
              for k,v in old.state_dict().items()) and
          not any(p.requires_grad for p in old.parameters()),
          "all 53,751 trained S115 original neural weights must remain exact")
    return losses

def run(s194,s115,s115_report,source,source_report,out):
    check(not out.exists() and source.name==PARENT_H and
          source_report.name==PARENT_REPORT and sha(source)==PARENT_H_SHA
          and sha(s115)==S115_EXPECTED_SHA256,
          "authentic SHA-locked old source-only full12 and trained S115 only")
    audit(s194,s115,s115_report)
    m=json.loads(source_report.read_text())
    check(m["ALL12_CPU_SAFETENSORS_SHA256"]==PARENT_H_SHA and
          m["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"] is True and
          m["source_ONLY_no_S113_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU"] is True and
          m["candidate_accepted"] is False,
          "genuine frozen Calibri+GT old 616 protected source proof missing")
    torch.manual_seed(119)
    torch.set_num_threads(2)
    phys=load_file(str(s115),device="cpu")
    c=UnifiedCalibri()
    c.load_original_s115(phys)
    f=load_file(str(source),device="cpu")
    h=f["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=f["native_BPE_source_valid_mask"]
    check(h.shape[0]==256 and h.shape[1]==12 and h.shape[-1]==320 and
          h.dtype==torch.float16 and valid.shape==(256,h.shape[2]) and
          valid.dtype==torch.bool,"true original12 source data geometry")
    n=int(valid[0].sum())
    check(2<=n<=h.shape[2] and bool(valid[0,:n].all()),
          "TRAIN-A first row native BPE valid")
    traces=train_joint(c,h[:1,:,:n].float().detach(),valid[:1,:n])
    states={"model.nra3_calibri_joint."+k:v.detach().half().contiguous()
            for k,v in c.state_dict().items()}
    check(sum(v.numel() for v in states.values())==84479,
          "full S115 53751 + NEW core 30720 + graph8, all in ONE model")
    for k,v in phys.items():
        u="model.nra3_calibri_joint."+k[len("model."):]
        check(torch.equal(states[u],v),"original S115 forgotten in unified child")
    out.mkdir(parents=True)
    saved=out/CHECKPOINT
    save_file(states,str(saved))
    read=load_file(str(saved),device="cpu")
    check(set(states)==set(read) and
          all(torch.equal(states[k],read[k]) for k in states),
          "physical unified checkpoint all old+new weights roundtrip")
    info={
        "stage":STAGE,"single_model_child":"model.model.nra3_calibri_joint",
        "source_old_CB59_GT_all12_SHA256":PARENT_H_SHA,
        "old_trained_S115_full53751_SHA256":S115_EXPECTED_SHA256,
        "new_trainable_core_rank4_parameters":30720,
        "new_graph_twohop_role_parameters":8,
        "entire_original_neural_NRA3_trainable_old_params_frozen":53751,
        "all_old_8_NRA2_mini_areas_included":True,
        "actual_original_train_A_source_rows_used":1,
        "real_native_BPE_count":n,
        "one_optimizer_and_one_loss_both_components_updated":True,
        "joint_gradient_trace":traces,
        "source_only_proxy_loss_NOT_semantic_gain":True,
        "original_full_decoder_NOT_connected_to_trial":True,
        "candidate_actual616_replay_NOT_run":True,
        "paired_CPU_speed_NOT_MEASURED":True,
        "all_in_physical_bpp_NOT_CERTIFIED":True,
        "SynTagRus_B_GOLD_NOT_LOADED":True,
        "noncommercial_CC_BY_NC_SA_4_0_research_only":True,
        "checkpoint_SHA256":sha(saved),
        "checkpoint_bytes":saved.stat().st_size,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False}
    (out/REPORT).write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    print("S119F_UNIFIED_ONE_CALIBRI_GRADIENT_PROOF "+
          json.dumps({"core_new_params":30720,"graph_new_params":8,
                      "old_neural_preserved":53751,
                      "source_tokens":n,"checkpoint_bytes":saved.stat().st_size,
                      "both_core_graph_trained":True,
                      "accepted":False},sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    for name in ("s194","s115","s115_report","source","source_report","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s194,a.s115,a.s115_report,a.source,a.source_report,a.out)

if __name__=="__main__":
    main()
