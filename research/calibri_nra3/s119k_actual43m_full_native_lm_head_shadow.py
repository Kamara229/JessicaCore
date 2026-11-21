"""S119k real original compressed Calibri 43M+GT full native LM-head shadow.

ACTUAL original 12-layer CB59/GT/S142 decoder is rebuilt from SHA-pinned
physical S93 CB61 + original S142 trained FP16, and run on original S114
source-only TRAIN-A native BPE prefixes. Original tokenizer and original
FULL vocabulary head score actual next-token IDs, not fake 162-token keys.
The all-old S115 53751 NRA3, S119g NPA1 and S119j structured bridge are
registered as ONE read-only-output-gated neural child of research Calibri.
A three-gate ZERO-initialized shadow head computes candidate logits OUTSIDE
the original decoder forward; the original decoder/GT never receives new
gradient or altered outputs. This is NOT full end-to-end original training,
genuine autoregressive generation, independent QA, exact 616 retention,
loader-complete bpp or matched CPU throughput certification.
No LIVE/Render/Supabase/Jessica/main, no S118-B GOLD, no QA leakage.
Third party TRAIN-A CC BY-NC-SA 4.0 noncommercial research ONLY.
"""
from __future__ import annotations
import argparse,hashlib,json,copy,time
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119j_real_native_bpe_target_context_ablation import (
    STAGE as PARENT_STAGE,CHILD as PARENT_FILE,REPORT as PARENT_REPORT)
from research.calibri_nra3.s119i_real12_contextual_structured_operator_ablation import (
    UnifiedCalibriWithStructuredInterface)
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import PREFIX
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    PARENT_H as SOURCE_FILE,PARENT_H_SHA as SOURCE_SHA,
    PARENT_REPORT as SOURCE_REPORT)

STAGE="S119K_TRUE_ORIGINAL43M_GT_S142_FULL_NATIVE_VOCAB_SHADOW_ONLY"
CHECKPOINT="S119K_ORIGINAL43M_GT_TRUE_FULL_NATIVE_HEAD_UNIFIED_SHADOW.safetensors"
REPORT="S119K_TRUE_NATIVE_FULL_VOCAB_SOURCE_TRAIN_ONLY_REPORT.json"
S142_SHA="5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7"
ROWS_TRAIN=(0,1);ROWS_PROBE=(128,129)
PREFIXES=(12,18);SEED=119124

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class NativeFullHeadShadow(nn.Module):
    def __init__(self):
        super().__init__()
        self.unified=UnifiedCalibriWithStructuredInterface()
        self.readout_gates=nn.Parameter(torch.zeros(3))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
    def load_original_child(self,physical):
        old={}
        for k,v in physical.items():
            check(k.startswith(PREFIX) and v.dtype==torch.float16,
                  "S119j actual FP16 unified model only")
            old[k[len(PREFIX):]]=v.float()
        check(set(old)==set(self.unified.state_dict())
              and sum(v.numel() for v in old.values())>116777,
              "no S119j original pretrained weights lost")
        self.unified.load_state_dict(old,strict=True)
        for p in self.unified.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        check(sum(p.numel() for p in self.unified.nra3_neural_weights.parameters())==53751,
              "full old in-weight NRA3 trainable 53751 preserved")
    def forward(self,h,last,head,*,contextual):
        check(self.output_gate==0 and not self.candidate_accepted and
              not self.activation_allowed and
              h.ndim==4 and h.shape[1]==12 and h.shape[3]==320 and
              h.shape[0]==1 and h.shape[2]>=2 and last.shape==(1,320),
              "one original Calibri true12 native prefix only; no online activation")
        valid=torch.ones((1,h.shape[2]),dtype=torch.bool)
        g,relations,plan,planned=self.unified(h,valid,contextual=contextual)
        old=self.unified.nra3_neural_weights(h,valid)
        adapted=self.unified.structured_interface(h[:,0],old,valid,
                                                  contextual=contextual)
        structure_delta=adapted[:,-1]-h[:,0,-1]
        core_delta=self.unified.core_growth(h)[:,-1,-1]-h[:,-1,-1]
        # Relational six-slot residual travels through same research readout.
        relation_six=F.pad(relations[:,-1],(0,314))
        gate=self.readout_gates.tanh()
        output=head(last+gate[0]*core_delta+
                    gate[1]*(structure_delta+.01*relation_six)+
                    gate[2]*planned)
        check(output.ndim==2 and output.shape[1]>1000
              and bool(torch.isfinite(output).all()),
              "ACTUAL full native original compressed LM head only")
        return output
    def trainable(self):
        return [*self.unified.new_joint_trainables(),self.readout_gates]

@contextmanager
def capture_all12(model):
    captured={};handles=[]
    check(len(model.model.layers)==12 and not model.training,
          "strict original compressed twelve-layer CPU model")
    def hook(i):
        def fn(_m,_x,out):
            x=out[0] if isinstance(out,tuple) else out
            check(i not in captured and
                  isinstance(x,torch.Tensor) and x.ndim==3 and
                  x.shape[-1]==320,"full original layer h320 hook once")
            captured[i]=x.detach().cpu().clone()
            return None
        return fn
    try:
        handles=[layer.register_forward_hook(hook(i))
                 for i,layer in enumerate(model.model.layers)]
        yield captured
        check(set(captured)==set(range(12)),
              "all 12 ORIGINAL layers must really execute")
    finally:
        for h in handles:h.remove()

def load_actual_original(s93,s93_side,s142):
    from calibri2.s87 import guard_all_456_full_cpu as s87
    from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
    from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
    from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
    from calibri2.s126.compact_rank4_q2 import ENCODED_KEYS,load_compact_q2
    from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
    manifest=json.loads(Path("research/calibri2/s144/source_manifest.json").read_text())
    check(s93.name=="S93_UNACCEPTED_SELECTED_CB61.s61" and
          s93_side.name=="S93_UNACCEPTED_SELECTED_JOINT.safetensors" and
          s142.name=="S142_ACTUAL_43M_ONE_J7_FP16.safetensors" and
          sha(s142)==S142_SHA and s142.stat().st_size==1001348 and
          manifest["checkpoint_sha256"]==S142_SHA,
          "exact original trained 43M sidecar SHA mismatch")
    side=load_file(str(s93_side),device="cpu")
    inherited=load_file(str(s142),device="cpu")
    check(set(inherited)==set(side)|set(ENCODED_KEYS)|{IDS_KEY,WEIGHTS_KEY}
          and all(torch.equal(inherited[k],v) for k,v in side.items())
          and inherited[WEIGHTS_KEY].shape==(12,320,48),
          "full S142 original S93+GT+grown core provenance")
    raw=s87.open_envelope(s93.read_bytes())
    cpu=load_cpu_original(raw,side)
    model,sources,*_=cpu
    bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(model)
    bridge.coefficients.requires_grad_(False)
    full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
    load_compact_q2(full,{k:inherited[k] for k in ENCODED_KEYS})
    full.install(model)
    expansion=ProgressiveRouteCore(rank=896).eval().bind_original_route_bridge(
        bridge).install(model)
    expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],
                                WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
    model.eval().requires_grad_(False)
    check(len(model.model.layers)==12,
          "only real S142 original12 neural core permitted")
    return model,sources,bridge,full,expansion

def original_parameter_hash(model):
    h=hashlib.sha256()
    for k,v in model.named_parameters():
        if "nra3_s119k_shadow" in k:continue
        h.update(k.encode()+b"\0"+
                 v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def real_cpu_prefix(model,sources,bridge,full,expansion,tok,ids,mask,row,prefix):
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from v176.operators import clear
    n=int(mask[row].sum())
    check(2<=prefix<n and prefix<=30 and bool(mask[row,:n].all())
          and not bool(mask[row,n:].any()),"actual native-BPE source prefix")
    prefix_ids=ids[row,:prefix].long().unsqueeze(0)
    next_id=int(ids[row,prefix])
    # S95 vector explicitly built from prefix-only, not the future source.
    txt=tok.decode(prefix_ids[0].tolist(),skip_special_tokens=False)
    vec=torch.as_tensor(s160.prompt_only_vector(tok,txt),dtype=torch.float32)
    check(vec.shape==(512,),"original S95 features prefix-only")
    with torch.inference_mode(),expansion.prefill_only(model),\
         s160.prefill_only_tunnel(model,full):
        with s160.frozen_cpu_phase_scope(model,bridge,"S93",vec,
                                          phase="prefill_only"):
            with capture_all12(model) as frames:
                clear(sources)
                out=model(input_ids=prefix_ids,
                          past_key_values=None,use_cache=True,
                          output_hidden_states=True)
                check(out.past_key_values is not None and
                      out.hidden_states[-1].shape==(1,prefix,320),
                      "original 43M GT real CPU prefill")
                final=out.hidden_states[-1][:,-1].detach().cpu().clone()
                native=out.logits[:,-1,:].float().detach().cpu().clone()
                manual=model.lm_head(final).float()
                check(manual.shape==native.shape and
                      torch.allclose(manual,native,rtol=1e-4,atol=2e-3),
                      "original fully native LM head must equal true model logits")
                del out
                clear(sources)
    # Outside inference_mode, copy makes ordinary differentiable inputs.
    full12=torch.stack([frames[k] for k in range(12)],1).float().clone()
    return {"h12":full12,"last":final.float().clone(),
            "base_logits":native.clone(),"next_id":next_id,
            "row":row,"prefix":prefix}

def full_vocab_ce(trial,ex,head,contextual):
    logits=trial(ex["h12"],ex["last"],head,contextual=contextual)
    target=torch.tensor([ex["next_id"]],dtype=torch.long)
    check(0<=ex["next_id"]<logits.shape[-1],
          "true original next-token BPE vocabulary ID")
    return F.cross_entropy(logits.float(),target),logits

def eval_trial(trial,examples,head,contextual):
    trial.eval()
    with torch.no_grad():
        rows=[]
        for x in examples:
            loss,logits=full_vocab_ce(trial,x,head,contextual)
            base=F.cross_entropy(x["base_logits"],
                                 torch.tensor([x["next_id"]]))
            rows.append({"source_row":x["row"],"prefix":x["prefix"],
                         "native_full_vocab_CE":float(loss),
                         "original_native_full_vocab_CE":float(base),
                         "native_full_vocab_top1":
                            int(int(logits.argmax(-1))==x["next_id"]),
                         "original_native_top1":
                            int(int(x["base_logits"].argmax(-1))==x["next_id"])})
    return {"mean_native_full_vocab_CE":sum(r["native_full_vocab_CE"]
                  for r in rows)/len(rows),
            "mean_original_full_vocab_CE":sum(
                 r["original_native_full_vocab_CE"] for r in rows)/len(rows),
            "native_full_vocab_top1_correct":sum(
                 r["native_full_vocab_top1"] for r in rows),
            "original_top1_correct":sum(r["original_native_top1"] for r in rows),
            "rows":rows}

def train(trial,examples,head,contextual):
    trial.train()
    opt=torch.optim.Adam(trial.trainable(),lr=.001)
    old={k:v.detach().half().clone()
         for k,v in trial.unified.nra3_neural_weights.state_dict().items()}
    logs=[]
    for epoch in range(2):
        for idx,x in enumerate(examples):
            opt.zero_grad(set_to_none=True)
            l,_=full_vocab_ce(trial,x,head,contextual)
            check(bool(torch.isfinite(l)),"full native vocabulary target CE")
            l.backward()
            check(trial.readout_gates.grad is not None and
                  bool(torch.isfinite(trial.readout_gates.grad).all()) and
                  float(trial.readout_gates.grad.abs().sum())>1e-10,
                  "true real native full LM head must train unified readout")
            if epoch==1 and idx==0:
                for name,param in [
                    ("core",trial.unified.core_growth.up),
                    ("structured",trial.unified.structured_interface.fuse.weight),
                    ("NPA",trial.unified.next_state_projector.weight),
                    ("graph",trial.unified.reasoning_role_gate)]:
                    g=param.grad
                    check(g is not None and bool(torch.isfinite(g).all()) and
                          float(g.abs().sum())>1e-14,
                          "full LM vocab same objective must train "+name)
            torch.nn.utils.clip_grad_norm_(trial.trainable(),1.)
            opt.step()
            logs.append({"epoch":epoch+1,"row":x["row"],"native_full_CE":float(l)})
    check(all(torch.equal(v.detach().half(),old[k]) for k,v in
              trial.unified.nra3_neural_weights.state_dict().items()),
          "all 53751 original learned NRA3 FP16 weights exactly retained")
    return logs

def run(s93,s93_side,s142,source,source_report,prior,prior_report,out):
    check(not out.exists() and source.name==SOURCE_FILE and
          source_report.name==SOURCE_REPORT and sha(source)==SOURCE_SHA and
          prior.name==PARENT_FILE and prior_report.name==PARENT_REPORT,
          "exact S114 original12 and fully trained S119j physical source")
    manifest=json.loads(source_report.read_text())
    parent=json.loads(prior_report.read_text())
    check(manifest["ALL12_CPU_SAFETENSORS_SHA256"]==SOURCE_SHA
          and manifest["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
          and manifest["source_ONLY_no_S113_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU"]
          and parent["stage"]==PARENT_STAGE
          and parent["physical_full_new_neural_child_SHA256"]==sha(prior)
          and parent["registered_unified_child_original_trained_NRA3_53751_exact"]
          and parent["candidate_accepted"] is False,
          "no original/J7/source-only provenance deviation")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    physical=load_file(str(prior),device="cpu")
    contextual=NativeFullHeadShadow()
    contextual.load_original_child(physical)
    local=copy.deepcopy(contextual)
    source_tensors=load_file(str(source),device="cpu")
    valid=source_tensors["native_BPE_source_valid_mask"]
    native=source_tensors["original_native_BPE_token_ids"]
    check(valid.shape==native.shape and valid.shape[0]==256 and
          valid.dtype==torch.bool and native.dtype==torch.int32,
          "full native original S114 tokenizer source without GOLD")
    real,sources,bridge,full,expansion=load_actual_original(
        s93,s93_side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    h0=original_parameter_hash(real)
    # Inert registered research neural module: real base decoder does not
    # call it. Only standalone full-native original-head shadow sees it.
    real.model.add_module("nra3_s119k_shadow",contextual)
    check(real.model.nra3_s119k_shadow is contextual and
          contextual.output_gate==0 and not contextual.candidate_accepted,
          "same real original Calibri owns research-only child")
    start=time.perf_counter()
    train_pairs=[real_cpu_prefix(real,sources,bridge,full,expansion,
                 tokenizer,native,valid,row,pref)
                 for row,pref in zip(ROWS_TRAIN,PREFIXES)]
    probe_pairs=[real_cpu_prefix(real,sources,bridge,full,expansion,
                 tokenizer,native,valid,row,pref)
                 for row,pref in zip(ROWS_PROBE,PREFIXES)]
    original_cpu_prefill_seconds=time.perf_counter()-start
    check(original_parameter_hash(real)==h0,
          "all real original trained 43M core/GT weights unchanged")
    head=real.lm_head
    before_c=eval_trial(contextual,probe_pairs,head,True)
    before_l=eval_trial(local,probe_pairs,head,False)
    check(before_c==before_l and abs(
          before_c["mean_native_full_vocab_CE"]-
          before_c["mean_original_full_vocab_CE"])<1e-5,
          "original full native LM head EXACT zero new-gate baseline")
    t1=train(contextual,train_pairs,head,True)
    t2=train(local,train_pairs,head,False)
    after_c=eval_trial(contextual,probe_pairs,head,True)
    after_l=eval_trial(local,probe_pairs,head,False)
    check(original_parameter_hash(real)==h0 and
          all(torch.equal(v,local.unified.nra3_neural_weights.state_dict()[k].detach().half())
              for k,v in {key:val.detach().half() for key,val in
               contextual.unified.nra3_neural_weights.state_dict().items()}.items()),
          "original core and 53751 old in-model learned NRA3 untouched")
    out.mkdir(parents=True)
    phys={PREFIX+"s119k_shadow."+k:v.detach().half().cpu().contiguous()
          for k,v in contextual.state_dict().items()}
    saved=out/CHECKPOINT
    save_file(phys,str(saved))
    reload=load_file(str(saved),device="cpu")
    check(set(phys)==set(reload) and all(torch.equal(phys[k],reload[k])
                                       for k in phys),
          "physical real original native-head child full FP16 roundtrip")
    full_vocab=int(probe_pairs[0]["base_logits"].shape[-1])
    report={
        "stage":STAGE,"real_original_s93_cb61_SHA256":sha(s93),
        "real_original_S93_side_sha256":sha(s93_side),
        "original_S142_43M_full_FP16_SHA256":sha(s142),
        "real_original_S114_ALL12_GT_SHA256":SOURCE_SHA,
        "previous_frozen_S119j_full_neural_sha256":sha(prior),
        "true_original12_CPU_prefills":len(train_pairs)+len(probe_pairs),
        "real_original12_frozen_GT_and_S142_core":True,
        "native_REAL_full_original_vocabulary_size":full_vocab,
        "REAL_original_full_native_LM_head_used":True,
        "target_is_TRUE_next_native_BPE_id_not_162class_proxy":True,
        "train_only_TRAIN_A_source_pairs":list(zip(ROWS_TRAIN,PREFIXES)),
        "probe_same_TRAIN_A_corpus_not_independent":list(zip(ROWS_PROBE,PREFIXES)),
        "baseline_identical_at_zero_gate":True,
        "base_original_native_full_vocab_probe":before_c,
        "trained_contextual_native_full_vocab_probe":after_c,
        "trained_equal_parameter_local_native_full_vocab_probe":after_l,
        "contextual_minus_local_CE":after_c["mean_native_full_vocab_CE"]-
                                     after_l["mean_native_full_vocab_CE"],
        "contextual_won_narrow_TRAIN_A_full_native_vocab_probe":
              after_c["mean_native_full_vocab_CE"]<
              after_l["mean_native_full_vocab_CE"],
        "contextual_train_trace":t1,"local_train_trace":t2,
        "real_original_CPU_prefill_total_seconds_NOT_throughput":
              original_cpu_prefill_seconds,
        "all_pretrained_old_NRA3_53751_FP16_exact":True,
        "real_original_43M_GT_s142_parameters_digest_before_after_equal":True,
        "physical_unified_shadow_FP16_weights_count":sum(x.numel()
                            for x in phys.values()),
        "physical_unified_shadow_bytes":saved.stat().st_size,
        "physical_unified_shadow_SHA256":sha(saved),
        "shadow_only_outside_original_decoder_forward":True,
        "new_components_NOT_fully_end_to_end_trained_through_original43M":True,
        "GT_prefix_streaming_causality_NOT_proven":True,
        "true_generated_universal_answers_NOT_tested":True,
        "original_616_generation_candidate_J7_NOT_REPLAYED":True,
        "new_full_model_CPU_throughput_NOT_MEASURED":True,
        "complete_all_in_bpp_NOT_MEASURED":True,
        "independent_unseen_qa_NOT_measured":True,
        "SynTagRusB_GOLD_not_opened":True,
        "license_CC_BY_NC_SA_4_noncommercial_research_only":True,
        "candidate_accepted":False,"activation_allowed":False,
        "output_gate":0,
        "decision":"ACTUAL_ORIGINAL43M_GT_NATIVE_FULL_VOCAB_SHADOW_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
        sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119K_REAL43M_GT_FULL_NATIVE_VOCAB "+
          json.dumps({"real_prefix_CPU":len(train_pairs)+len(probe_pairs),
                      "full_original_vocab":full_vocab,
                      "baseline_CE":before_c["mean_native_full_vocab_CE"],
                      "contextual_CE":after_c["mean_native_full_vocab_CE"],
                      "matched_local_CE":after_l["mean_native_full_vocab_CE"],
                      "contextual_won":report[
                          "contextual_won_narrow_TRAIN_A_full_native_vocab_probe"],
                      "old_NRA3_preserved":True,
                      "candidate_accepted":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("s93","s93_side","s142","source","source_report",
              "prior","prior_report","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.s93_side,a.s142,a.source,a.source_report,
        a.prior,a.prior_report,a.out)

if __name__=="__main__":main()
