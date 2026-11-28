"""Native teacher-forced single traversal, positionwise NRA3, no weight change."""
import torch
from research.calibri2.fusion_growth.f06_integrated_recovery import scoped_reduced_native_head_gain
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import temporary_learned_native_head
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from .position_hooks import causal_positions

def native_positions(st,tokens,features,slot,l6,balance,positions,inject=None):
    model,sources,bridge,full,expansion,child,_=st
    device=next(model.parameters()).device
    assert positions and max(positions)<len(tokens)
    inject=tuple(positions if inject is None else inject)
    index=torch.tensor(positions,dtype=torch.long,device=device)
    ids=torch.tensor([tokens],dtype=torch.long,device=device);cap=[]
    h1=model.lm_head.register_forward_pre_hook(lambda m,a:cap.append(a[0].index_select(1,index)))
    try:
        with torch.enable_grad(),scoped_reduced_native_head_gain(child.language_head),temporary_learned_native_head(model,child.language_head) as audit,expansion.prefill_only(model),s160.prefill_only_tunnel(model,full),bridge.prompt_scope("S93",features),causal_positions(st,l6,balance,inject) as witness:
            clear(sources)
            try:
                out=model(input_ids=ids,past_key_values=None,use_cache=True,output_hidden_states=False)
                z=out.logits.index_select(1,index).float()
                assert out.past_key_values is not None
            finally:clear(sources)
        assert audit["original_head_calls"]==witness["core_mid_calls"]==len(cap)==1
        assert witness["causal_prefix_calls"]==len(inject) and full.last_forward_coverage==tuple(range(12))
        with scoped_reduced_native_head_gain(child.language_head):
            s=slot(cap[0]);z2=(model.lm_head(s).float()+child.language_head(s).float())
        clear(sources)
        assert z.shape==z2.shape==(1,len(positions),49152)
        return z[0],z2[0]
    finally:h1.remove()
