"""Z68 component localization on exactly 57 incomplete TRAIN42 prefixes.
No training, DEV, FP32 or checkpoint selection. Same saved Z55 GT12 gates.
Separate base compressed decoder, on-path child, and native head correction.
"""
import torch
from collections import Counter
from research.calibri_nra3.s119q2i8z63_utf8_transition_objective import byte_pieces,advance,allowed_mask
ARMS=("ORIGINAL_NO_CHILD","Z55_CHILD_HEAD_ZERO","Z55_CHILD_HEAD_025")
def check(ok,msg):
    if not ok:raise ValueError(msg)
def collect(tok,rows):
    pieces=byte_pieces(tok);cache={};out=[]
    for row in rows:
        gold=row["answer_native_with_eos"];pending=b""
        for t,target in enumerate(gold):
            if pending:
                if pending not in cache:cache[pending]=allowed_mask(pieces,pending,tok.eos_token_id)
                mask=cache[pending]
                check(bool(mask[target]) and not bool(mask[tok.eos_token_id]),"actual valid pending target")
                out.append({"id":row["id"],"kind":row["kind"],"position":t+1,
                       "prefix":row["prefix"]+gold[:t],"features":row["prompt_features"],
                       "target":target,"mask":mask,"pending":pending.hex()})
            if target==tok.eos_token_id:check(not pending and t==len(gold)-1,"true terminal EOS")
            else:
                pending=advance(pending,pieces[target]);check(pending is not None,"valid authored bytes")
    check(len(out)==57 and len(rows)==42,"exact Z58 incomplete reference positions")
    return out
def measure(logits,target,mask):
    check(logits.shape==(1,mask.numel()) and not logits.requires_grad and bool(torch.isfinite(logits).all()),"frozen native logits")
    x=logits[0].float();chosen=int(x.argmax());lp=torch.log_softmax(x,dim=0)
    return {"top1":chosen,"gold_rank":1+int((x>x[target]).sum()),"correct":chosen==target,
          "valid_top1":bool(mask[chosen]),"valid_mass":float(torch.logsumexp(lp[mask],dim=0).exp()),
          "gold_NLL":-float(lp[target])}
def summarize(records):
    check(len(records)==57,"all 57 incomplete reference positions")
    return {"n":len(records),"valid_top1":sum(r["valid_top1"] for r in records),
            "exact_top1":sum(r["correct"] for r in records),
            "mean_valid_mass":sum(r["valid_mass"] for r in records)/len(records),
            "mean_gold_NLL":sum(r["gold_NLL"] for r in records)/len(records),
            "mean_gold_rank":sum(r["gold_rank"] for r in records)/len(records),
            "top_invalid_ids":Counter(r["top1"] for r in records if not r["valid_top1"]).most_common(10)}
