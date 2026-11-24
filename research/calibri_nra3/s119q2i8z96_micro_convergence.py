"""Z96 fixed-budget extension of Z94 small-data experiment, never deploy.
All training on 12 existing GPT-authored definition examples, no replay.
No claims of generalization or preservation of language ability.
"""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import prepare as prepare_all,PARENT_SHA,GT_SHA,verify_z73
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import scoped_reduced_native_head_gain
GAIN=.25
STEPS=600
def prepare(tok):
    _,_,new,authored=prepare_all(tok)
    rows=[r for r in new if authored[r["id"]]["family"]=="definition"]
    check(len(rows)==12 and sum(len(r["answer_native_with_eos"]) for r in rows)==51,"exact twelve definitions, native targets")
    return rows,authored
def cache_rows(model,sources,bridge,full,expansion,child,rows):
    # The raw LM head hook runs BEFORE temporary learned correction.
    # Cache actual frozen compressed-model hidden + native base logits.
    # Historical physical forward contract requires a differentiable
    # one-pass child. Retain the standard research requires_grad flags
    # only while capturing; never backpropagate or update during cache.
    # All captured tensors are detached, and actual graph is discarded.
    from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import make_joint_trainable
    selected=make_joint_trainable(child)
    for p in full.parameters():p.requires_grad_(False)
    check(sum(p.numel() for p in selected)==368039,"same historical differentiable child contract")
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
                del actual,reconstructed
                check(all(p.grad is None for p in selected),"capture only, no backward or weight updates")
            finally:hook.remove()
        result.append({"id":row["id"],"gold":row["answer_native_with_eos"],
                       "hidden":torch.cat(hidden,dim=0),"base":torch.cat(base,dim=0),
                       "masks":row["utf8_masks"]})
    for p in child.parameters():p.requires_grad_(False)
    check(len(result)==12 and count==51,"exact frozen definition TRAIN12 cache")
    return result,{"real_compressed_decoder_forwards":count,"max_reconstruction_abs_error":max_error,
                   "all_positions_top1_equal":True}


def batch(child,cache):
    h=torch.cat([e["hidden"] for e in cache])
    b=torch.cat([e["base"] for e in cache])
    gold=torch.tensor([g for e in cache for g in e["gold"]],dtype=torch.long)
    check(h.shape==(51,1,320) and b.shape==(51,49152) and gold.shape==(51,),"real frozen data")
    return h,b,gold
def logits_for(head,h,b):
    return b+GAIN*head(h)[:,0,:]
def metrics(logits,gold):
    return {"NLL":float(F.cross_entropy(logits,gold)),"top1":int((logits.argmax(1)==gold).sum()),
        "positions":len(gold),"mean_gold_rank":float(((logits>logits.gather(1,gold[:,None])).sum(1)+1).float().mean())}
def train_cached(child,cache):
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==247040,"same existing head")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    h,b,gold=batch(child,cache)
    opt=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.003},{"params":[head.output_bias],"lr":.001}])
    history=[]
    for i in range(STEPS+1):
        if i in (0,10,30,60,120,240,360,480,600):
            with torch.no_grad():m=metrics(logits_for(head,h,b),gold)
            history.append({"step":i,**m});print("Z96_MICRO_PROGRESS",history[-1],flush=True)
            if i==120:
                check(abs(m["NLL"]-2.259514570236206)<.03 and abs(m["top1"]-16)<=2,
                      "same Z94 learning trajectory before extending fixed budget")
        if i==STEPS:break
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(logits_for(head,h,b),gold)
        check(bool(torch.isfinite(loss)),"finite full vocab CE")
        loss.backward();torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True);opt.step()
        check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite actual learned head")
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(all(v>0 for v in changed.values()) and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),"real head training, frozen body")
    return {"Adam_updates":STEPS,"cached_training_positions":STEPS*51,"history":history,"head_change_L2":changed,"new_parameters":0}
def own_metrics(records):
    check(len(records)==12 and len({r["id"] for r in records})==12,"exact TRAIN12")
    return {"n":12,**{k:sum(r[k] for r in records) for k in ("complete_correct","text_correct","EOS","valid_UTF8")},
            "first_two_correct":sum(r["prefix_match"]>=2 for r in records),"forwards":sum(len(r["generated_ids"]) for r in records)}
