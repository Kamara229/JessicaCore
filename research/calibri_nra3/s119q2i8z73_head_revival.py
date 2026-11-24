"""Z73 matched four-epoch test of function-preserving revival of verified
inactive existing rank4 units. No new parameters or inference changes.
Zero output columns preserve initial native logits before learning.
"""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z70_cached_head_train import cache_rows,row_loss,ARMS as BASE_ARMS,GAIN
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import HEAD_PARAMETER_COUNT
from research.calibri_nra3.s119a_hard_four_goals import check
ARMS=("HEAD_CONTROL","HEAD_REVIVE_ZERO_UNITS")
EPOCHS=4
@torch.no_grad()
def activation_summary(head,cache):
    h=torch.cat([r["hidden"] for r in cache],0)
    z=F.gelu(head.down(F.layer_norm(h,(320,))))[:,0,:]
    check(z.shape==(sum(len(r["gold"]) for r in cache),4) and bool(torch.isfinite(z).all()),"actual rank4 activations")
    maximum=z.abs().amax(0)
    s=torch.linalg.svdvals(z.double()-z.double().mean(0));q=s.square();p=q/q.sum().clamp_min(1e-30);p=p[p>0]
    return {"max_abs_activation":maximum.tolist(),"exact_zero_units":[i for i,v in enumerate(maximum) if float(v)==0.],
            "singular_values":s.tolist(),"effective_rank":float(torch.exp(-(p*p.log()).sum()))}
@torch.no_grad()
def revive(head,cache):
    before=activation_summary(head,cache)
    dead=before["exact_zero_units"]
    check(0<len(dead)<4,"verified inactive units on all cached TRAIN, else no experiment")
    old_down=head.down.weight.detach().clone();old_up=head.up.weight.detach().clone()
    old_bias=head.output_bias.detach().clone()
    active=[i for i in range(4) if i not in dead]
    g=torch.Generator(device="cpu").manual_seed(119173)
    # Deterministic fresh directions orthogonal to span of preserved active rows.
    basis=[]
    for i in active:
        v=old_down[i].double().clone()
        for q in basis:v-=torch.dot(v,q)*q
        if float(v.norm())>1e-10:basis.append(v/v.norm())
    for i in dead:
        v=torch.randn(320,generator=g,dtype=torch.float64)
        for q in basis:v-=torch.dot(v,q)*q
        check(float(v.norm())>1e-10,"nonzero independent direction")
        v=v/v.norm();basis.append(v)
        head.down.weight[i].copy_(.1*v.to(head.down.weight.dtype))
        head.up.weight[:,i].zero_()
    maxdiff=0.
    for r in cache:
        h=r["hidden"];z=F.gelu(F.linear(F.layer_norm(h,(320,)),old_down))
        old=r["base"]+GAIN*(F.linear(z,old_up)+old_bias)[:,0,:]
        new=r["base"]+GAIN*head(h)[:,0,:]
        maxdiff=max(maxdiff,float((old-new).abs().max()))
        check(torch.allclose(old,new,atol=2e-4,rtol=1e-5) and torch.equal(old.argmax(-1),new.argmax(-1)),
              "initial output parity on all actual cached native positions")
    check(torch.equal(head.down.weight[active],old_down[active]) and torch.equal(head.up.weight[:,active],old_up[:,active])
          and torch.equal(head.output_bias,old_bias),"preserve active directions and bias")
    after=activation_summary(head,cache)
    check(not set(dead).intersection(after["exact_zero_units"]),"revived hidden directions nonzero")
    return {"dead_units":dead,"reinitialized":True,"new_down_norm":.1,"initial_max_native_logit_difference":maxdiff,
            "initial_top1_parity":True,"activation_after_initialization":after}
def train_cached(child,cache,contrasts,*,arm):
    check(arm in ARMS,"fixed revival control and treatment")
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==HEAD_PARAMETER_COUNT==247040,
          "only existing rank4 native head")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    state_before=activation_summary(head,cache)
    revival=revive(head,cache) if arm==ARMS[1] else {"dead_units":state_before["exact_zero_units"],"reinitialized":False}
    check(bool(state_before["exact_zero_units"]),"must verify actual dead units on full TRAIN98, not infer from SVD alone")
    optimizer=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.0008},
                                {"params":[head.output_bias],"lr":.000025}])
    epochs=[];updates=0;positions=0
    for epoch in range(EPOCHS):
        nll=0.;aux=0.;n=0
        for entry in cache:
            optimizer.zero_grad(set_to_none=True)
            loss,ce,extra=row_loss(head,entry,contrasts.get(entry["id"]),BASE_ARMS[0])
            check(bool(torch.isfinite(loss)),"finite actual cached head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True)
            optimizer.step();updates+=1
            check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite physical head updates")
            n+=len(entry["gold"]);nll+=ce;aux+=extra
        check(n==1618,"same complete TRAIN exposure each epoch")
        positions+=n
        epochs.append({"epoch":epoch+1,"cached_native_positions":n,"mean_online_NLL":nll/n,"aux_loss_sum":aux})
        print("Z73_EPOCH",arm,epochs[-1],flush=True)
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(updates==392 and positions==6472 and all(v>0 for v in changed.values())
          and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),
          "physical head tensors changed; other weights bitwise preserved")
    return {"arm":arm,"Adam_updates":updates,"cached_training_positions":positions,
            "trainable_parameters":HEAD_PARAMETER_COUNT,"head_change_L2":changed,"epochs":epochs,
            "activation_before":state_before,"activation_after":activation_summary(head,cache),"revival":revival,"pending_UTF8_multiplier":1,"new_parameters":0,"FP32_source_calls":0}
