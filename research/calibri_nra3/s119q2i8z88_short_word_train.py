"""Z88 first physical shared-word curriculum pilot, before/after only.
All previous TRAIN98 rows retained. Same learned native head, no new params.
"""
import torch
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer,real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import scoped_reduced_native_head_gain
from research.calibri_nra3.s119q2i8z65_stable_gate_train import prepare as old_prepare
from research.calibri_nra3.s119q2i8z63_utf8_transition_objective import byte_pieces,advance,allowed_mask
from research.calibri_nra3.s119q2i8z87_short_word_curriculum import native,fresh
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import row_loss,ARMS,PARENT_SHA,GT_SHA
from research.calibri_nra3.s119q2i8z75_joint_after_head import verify_z73
from research.calibri_nra3.s119q2i8z73_head_revival import activation_summary
GAIN=.25
EPOCHS=4
NATIVE_SHA="fbaa6959b5d3283a6e865b786134da346837df4bb6528766b6ebb7af6e3a960b"
FRESH_SHA="f542727442812c48e66016d572fcc03a35db2958c28c4a58c18da24480ba8dd0"
def prepare(tok):
    proof=native(tok)
    check(proof["native_plan_SHA256"]==NATIVE_SHA and proof["new_SHA256"]==FRESH_SHA
          and proof["native_target_positions"]==204 and proof["pending_positions"]==8,"exact successful Z87 prereg")
    old,contrasts=old_prepare(tok)
    authored=fresh();new=[verified_native_answer(tok,r) for r in authored]
    pieces=byte_pieces(tok);masks={}
    for row in new:
        state=b"";row["utf8_masks"]=[]
        for g in row["answer_native_with_eos"]:
            if state not in masks:masks[state]=allowed_mask(pieces,state,tok.eos_token_id)
            m=masks[state];check(bool(m[g]),"legal authored target")
            row["utf8_masks"].append(m)
            if g!=tok.eos_token_id:state=advance(state,pieces[g]);check(state is not None,"valid byte state")
            else:check(not state,"terminal EOS")
    # Fixed interleave: approximately two replay rows per new short row.
    rows=[]
    for i,r in enumerate(new):
        rows+=old[2*i:2*i+2]+[r]
    rows+=old[96:]
    check(len(rows)==146 and len({r["id"] for r in rows})==146 and sum(len(r["answer_native_with_eos"]) for r in rows)==1822,"all replay and new positions")
    return rows,contrasts,new,{r["id"]:r for r in authored}
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
    check(len(result)==146 and count==1822,"exact frozen TRAIN146 cache")
    return result,{"real_compressed_decoder_forwards":count,"max_reconstruction_abs_error":max_error,
                   "all_positions_top1_equal":True}

def train_cached(child,cache,contrasts):
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==247040,"same existing head")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.0008},{"params":[head.output_bias],"lr":.000025}])
    epochs=[];updates=0
    for e in range(EPOCHS):
        stats={"short":{"rows":0,"positions":0,"CE_sum":0.},"replay":{"rows":0,"positions":0,"CE_sum":0.}}
        for entry in cache:
            optimizer.zero_grad(set_to_none=True)
            loss,m=row_loss(head,entry,contrasts.get(entry["id"]),ARMS[0])
            check(bool(torch.isfinite(loss)),"finite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True)
            optimizer.step();updates+=1
            check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite learned tensors")
            a=stats["short" if entry["id"].startswith("S119Q2I8Z87_") else "replay"]
            a["rows"]+=1;a["positions"]+=len(entry["gold"]);a["CE_sum"]+=m["CE_sum"]
        check(stats["short"]["rows"]==48 and stats["short"]["positions"]==204 and stats["replay"]["rows"]==98 and stats["replay"]["positions"]==1618,"all examples each epoch")
        for a in stats.values():a["mean_online_NLL"]=a["CE_sum"]/a["positions"]
        epochs.append({"epoch":e+1,**stats});print("Z88_EPOCH",epochs[-1],flush=True)
    check(updates==584 and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),"exact training budget and frozen body")
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(all(v>0 for v in changed.values()),"actual physical head updates")
    return {"Adam_updates":updates,"cached_training_positions":7288,"epochs":epochs,"head_change_L2":changed,"activation_after":activation_summary(head,cache),"FP32_calls":0,"new_parameters":0}
def short_metrics(records):
    check(len(records)==48 and len({r["id"] for r in records})==48,"complete known word prompts")
    def sub(rs):
        return {"n":len(rs),"correct":sum(r["complete_correct"] for r in rs),"text_correct":sum(r["text_correct"] for r in rs),
                "valid_UTF8":sum(r["valid_UTF8"] for r in rs),"EOS":sum(r["EOS"] for r in rs),
                "first_two_correct":sum(r["prefix_match"]>=2 for r in rs),"forwards":sum(len(r["generated_ids"]) for r in rs)}
    return {**sub(records),"by_family":{f:sub([r for r in records if r["family"]==f]) for f in sorted({r["family"] for r in records})}}
