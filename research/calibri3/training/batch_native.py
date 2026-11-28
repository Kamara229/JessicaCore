"""Actual B x T decoder pass, same-length prefixes, no padding."""
import torch
from research.calibri2.fusion_growth.f06_integrated_recovery import scoped_reduced_native_head_gain
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import temporary_learned_native_head
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from .batch_hooks import causal_batch

def native_batch(st,prefixes,features,slot,l6,balance):
    model,sources,bridge,full,expansion,child,_=st
    n=len(prefixes);assert n>=2 and len({len(p) for p in prefixes})==1
    # In this exact checkpoint S111 bridge is a frozen zero no-op.
    # If not, refuse to share a single context vector across rows.
    assert not bridge.coefficients.requires_grad
    assert not bool(torch.count_nonzero(bridge.coefficients.detach())), "nonzero S111 requires true per-row feature scope"
    device=next(model.parameters()).device
    ids=torch.tensor(prefixes,dtype=torch.long,device=device);cap=[];calls=[]
    h1=model.lm_head.register_forward_pre_hook(lambda _m,a:cap.append(a[0][:,-1:,:]))
    h2=model.register_forward_pre_hook(lambda *_:calls.append(1))
    try:
        with torch.enable_grad(),scoped_reduced_native_head_gain(child.language_head),temporary_learned_native_head(model,child.language_head) as audit,expansion.prefill_only(model),s160.prefill_only_tunnel(model,full),bridge.prompt_scope("S93",features[0]),causal_batch(st,l6,balance,n) as witness:
            clear(sources)
            try:
                out=model(input_ids=ids,past_key_values=None,use_cache=True,output_hidden_states=False)
                z=out.logits[:,-1,:].float()
                assert out.past_key_values is not None
            finally:clear(sources)
        assert audit["original_head_calls"]==witness["core_mid_calls"]==len(cap)==len(calls)==1
        assert witness["nra3_row_calls"]==n and full.last_forward_coverage==tuple(range(12))
        with scoped_reduced_native_head_gain(child.language_head):
            s=slot(cap[0]);z2=(model.lm_head(s).float()+child.language_head(s).float())[:,0,:]
        clear(sources)
        assert z.shape==z2.shape==(n,49152)
        return z,z2
    finally:h1.remove();h2.remove()
