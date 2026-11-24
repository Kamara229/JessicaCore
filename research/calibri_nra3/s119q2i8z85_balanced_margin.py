"""Z85 bounded state-balanced training-only margin after negative Z79.
Use all reference positions, explicitly penalize illegal bytes AND early EOS
at complete and pending states, leave terminal EOS CE unchanged.
No inference constraint or new model weights. Preflight only.
"""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import ARMS as BASE_ARMS,row_loss as baseline_loss,PARENT_SHA,GT_SHA
ARMS=("REVIVED_BASE","REVIVED_BALANCED_STATE_MARGIN")
TRAIN_ROWS=98
PENDING=123
COMPLETE_NONTERMINAL=1397
MARGIN=.5
WEIGHT=.1
def state_margin(logits,target,mask):
    if logits.ndim!=2 or logits.shape[0]!=1 or mask.shape!=(logits.shape[1],) or mask.dtype!=torch.bool:
        raise ValueError("full native logits and bool mask")
    if not 0<=target<mask.numel() or not bool(mask[target]):
        raise ValueError("legal gold required")
    if target==2:
        return logits.new_zeros(())
    pending=not bool(mask[2])
    bad=~mask
    bad[2]=True
    if bool(bad[target]):raise ValueError("gold must not be penalized")
    x=logits[0].float()
    denom=PENDING if pending else COMPLETE_NONTERMINAL
    # Average row objective over 98 rows gives equal .1 aggregate
    # coefficient to each state group; no per-answer rare-state boost.
    return WEIGHT*TRAIN_ROWS/denom*F.softplus(x[bad].max()-x[target]+MARGIN)
def row_loss(head,entry,contrast,arm):
    if arm not in ARMS:raise ValueError("fixed arms only")
    base,metrics=baseline_loss(head,entry,contrast,BASE_ARMS[0])
    gold=entry["gold"]
    logits=entry["base"]+.25*head(entry["hidden"])[:,0,:]
    if len(gold)!=len(entry["masks"]) or gold[-1]!=2:raise ValueError("exact reference sequence")
    pending=complete=0
    a=base.new_zeros(());b=base.new_zeros(())
    for t,(g,m) in enumerate(zip(gold,entry["masks"])):
        if g==2:
            if t!=len(gold)-1 or not bool(m[2]):raise ValueError("terminal complete EOS")
            continue
        loss=state_margin(logits[t:t+1],g,m) if arm==ARMS[1] else base.new_zeros(())
        if bool(m[2]):complete+=1;b=b+loss
        else:pending+=1;a=a+loss
    return base+a+b,{**metrics,"pending_margin":float(a.detach()),"complete_margin":float(b.detach()),
                    "pending_positions":pending,"complete_nonterminal_positions":complete}
def manifest():
    return {"stage":"Z85_BALANCED_STATE_MARGIN_PREFLIGHT","arms":ARMS,"parent_joint_SHA256":PARENT_SHA,
            "GT12_SHA256":GT_SHA,"TRAIN_rows":TRAIN_ROWS,"pending_positions":PENDING,
            "complete_nonterminal_positions":COMPLETE_NONTERMINAL,"terminal_EOS_positions":98,
            "total_positions":1618,"margin":MARGIN,"weight_each_state_group":WEIGHT,
            "terminal_CE_unchanged":True,"early_EOS_is_prohibited_competitor":True,
            "Z79_negative_control_run":45469297006,"no_FP32":True,"no_DEV":True,"no_inference_masks":True,
            "preflight_only":True,"candidate_accepted":False,"output_gate":0}
if __name__=="__main__":
    import json
    print("Z85_PREREG",json.dumps(manifest(),sort_keys=True),flush=True)
