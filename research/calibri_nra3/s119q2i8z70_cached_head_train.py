"""Z70 exact frozen-native readout caching and matched eight-epoch training.
Only existing rank4 output correction parameters update. No FP32 source
model, no external teacher, no DEV labels, no inference masking.
"""
import copy
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z65_stable_gate_train import objective_loss,ARMS as JOINT_ARMS
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import scoped_reduced_native_head_gain
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native,HEAD_PARAMETER_COUNT
from research.calibri_nra3.s119a_hard_four_goals import check
ARMS=("HEAD_UTF8_BASE","HEAD_UTF8_PENDING_X4")
EPOCHS=8
GAIN=.25
def cache_rows(model,sources,bridge,full,expansion,child,rows):
    # The raw LM head hook runs BEFORE temporary learned correction.
    # Cache actual frozen compressed-model hidden + native base logits.
    for p in child.parameters():p.requires_grad_(False)
    for p in full.parameters():p.requires_grad_(False)
    for p in child.language_head.parameters():p.requires_grad_(True)
    count=0;max_error=0.;result=[]
    for row in rows:
        hidden=[];base=[]
        for t,target in enumerate(row["answer_native_with_eos"]):
            captured=[]
            def capture(_module,args,output):
                check(not captured and args[0].shape[-1]==320 and output.shape[-1]==49152,
                      "actual single original native head")
                captured.append((args[0][:,-1:,:].detach().float().cpu().clone(),
                                 output[:,-1,:].detach().float().cpu().clone()))
            hook=model.lm_head.register_forward_hook(capture,prepend=True)
            try:
                with scoped_reduced_native_head_gain(child.language_head) as witness:
                    actual=real_joint_native(model,sources,bridge,full,expansion,child,
                        row["prefix"]+row["answer_native_with_eos"][:t],row["prompt_features"],contextual=True)
                    check(witness[0]==1 and len(captured)==1 and full.last_forward_coverage==tuple(range(12)),
                          "one real full compressed decoder per cache item")
                h,b=captured[0]
                with torch.no_grad():reconstructed=b+GAIN*child.language_head(h)[:,0,:]
                actual=actual.detach().float().cpu()
                e=float((reconstructed-actual).abs().max())
                check(torch.allclose(reconstructed,actual,atol=2e-4,rtol=1e-5)
                      and int(reconstructed.argmax())==int(actual.argmax()),
                      "exact cached native head forward parity")
                max_error=max(max_error,e);hidden.append(h);base.append(b);count+=1
            finally:hook.remove()
        result.append({"id":row["id"],"gold":row["answer_native_with_eos"],
                       "hidden":torch.cat(hidden,dim=0),"base":torch.cat(base,dim=0),
                       "masks":row["utf8_masks"]})
    check(len(result)==98 and count==1618,"exact frozen TRAIN98 cache")
    return result,{"real_compressed_decoder_forwards":count,"max_reconstruction_abs_error":max_error,
                   "all_positions_top1_equal":True}
def row_loss(head,entry,contrast,arm):
    check(arm in ARMS,"fixed matched arm")
    gold=entry["gold"]
    logits=entry["base"]+GAIN*head(entry["hidden"])[:,0,:]
    losses=[];ce_sum=0.;aux_sum=0.
    for t in range(len(gold)):
        m=entry["masks"][t]
        total,ce,pair,eos,utf8=objective_loss(logits[t:t+1],gold,t,arm=JOINT_ARMS[1],contrast=contrast,mask=m)
        if arm==ARMS[1] and not bool(m[2]):total=total+3.*utf8
        losses.append(total)
        ce_sum+=float(ce.detach())*len(gold)
        aux_sum+=float((total-ce).detach())
    return torch.stack(losses).sum(),ce_sum,aux_sum
def train_cached(child,cache,contrasts,*,arm):
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==HEAD_PARAMETER_COUNT==247040,
          "only existing rank4 native head")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.0008},
                                {"params":[head.output_bias],"lr":.000025}])
    epochs=[];updates=0;positions=0
    for epoch in range(EPOCHS):
        nll=0.;aux=0.;n=0
        for entry in cache:
            optimizer.zero_grad(set_to_none=True)
            loss,ce,extra=row_loss(head,entry,contrasts.get(entry["id"]),arm)
            check(bool(torch.isfinite(loss)),"finite actual cached head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True)
            optimizer.step();updates+=1
            check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite physical head updates")
            n+=len(entry["gold"]);nll+=ce;aux+=extra
        check(n==1618,"same complete TRAIN exposure each epoch")
        positions+=n
        epochs.append({"epoch":epoch+1,"cached_native_positions":n,"mean_online_NLL":nll/n,"aux_loss_sum":aux})
        print("Z70_EPOCH",arm,epochs[-1],flush=True)
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(updates==784 and positions==12944 and all(v>0 for v in changed.values())
          and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),
          "physical head tensors changed; other weights bitwise preserved")
    return {"arm":arm,"Adam_updates":updates,"cached_training_positions":positions,
            "trainable_parameters":HEAD_PARAMETER_COUNT,"head_change_L2":changed,"epochs":epochs,
            "pending_UTF8_multiplier":4 if arm==ARMS[1] else 1,"new_parameters":0,"FP32_source_calls":0}
