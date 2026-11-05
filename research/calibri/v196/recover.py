"""Green Tunnel recovery coordinator: simultaneous 211-codebook training over cycles.

Already-paid FP16 centroids updated in ALL packed operators on every
step. One registered paid 5-bit matrix receives source-gradient code
proposals each step. Start from authentic trained v186, not from
uncompressed teacher or the earlier nonfunctional CB59 baseline.
Never use DEV/VALID answers to optimize or choose stopping checkpoints.
"""
import json
import math
import torch

from v176.operators import clear
from v185.optim import make_student,propose_codes,apply_codes
from v174.run import save
from . import protocol as P
from .policy import POLICY
from .objective import choose_objective


def build_trainable_from_paid_source(archive):
    student,sources=make_student(archive)
    if len(sources)!=P.CODEBOOK_MATRIX_COUNT:
        raise AssertionError("Recovery omitted original 135M layers")
    if any(src.centroids.shape!=(32,8) for src in sources.values()):
        raise AssertionError("Unexpected not-paid codebook geometry")
    if student.model.embed_tokens.source is not student.lm_head.source:
        raise AssertionError("Green Tunnel broke tied embedding/LM head")
    total_grad_params=sum(x.numel() for x in student.parameters() if x.requires_grad)
    if total_grad_params!=P.CODEBOOK_MATRIX_COUNT*32*8:
        raise AssertionError("Unpaid extra trainable Green Tunnel neural sidecar")
    return student,sources


def recover(student,sources,train_teacher_refs,train_pairs,original_fp32_matrices,out):
    if len(train_teacher_refs)!=len(P.PUBLIC_TRAIN_INDICES) or len(train_pairs)!=P.TRAIN_PAIRS:
        raise AssertionError("Frozen public-training completeness failed")
    if len(sources)!=P.CODEBOOK_MATRIX_COUNT:
        raise AssertionError("Not restoring every source layer")
    expected=set(P.ORIGINAL_LAYER_SCHEDULE)
    if set(original_fp32_matrices)!=expected or not expected<=set(sources):
        raise AssertionError("Unpaid source operator/teacher proposal mismatch")
    optimizer=torch.optim.Adam(
        [x.centroids for x in sources.values()],lr=POLICY.learning_rate)
    original_centroids={name:x.centroids.detach().clone() for name,x in sources.items()}
    original_codes={name:x.codes.detach().clone() for name,x in sources.items()}
    records=[]
    total_proposed=0;total_accepted=0
    total_steps=POLICY.cycles*POLICY.steps_per_cycle
    assert total_steps==192
    for step in range(total_steps):
        target_name=P.ORIGINAL_LAYER_SCHEDULE[step%POLICY.steps_per_cycle]
        target=sources[target_name]
        captured=[]
        def capture(_source,_args,reconstructed):
            if not reconstructed.requires_grad:
                raise AssertionError("No end-to-end trainable source reconstruction")
            # Balanced-pair TRAIN makes TWO forwards through one cached
            # paid source tensor. The module hook fires twice, but both
            # visits share one reconstruction/gradient; deduplicate by
            # tensor ID as in the already validated original v186 fit.
            if not any(reconstructed is earlier for earlier in captured):
                reconstructed.retain_grad()
                captured.append(reconstructed)
        hook=target.register_forward_hook(capture)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            objective,stats,kind,index=choose_objective(
                student,step,train_teacher_refs,train_pairs)
            # Small all-layer reconstruction prior *only* from the
            # initial physical v186 paid centroids, not DEV answers.
            anchor=torch.stack([torch.mean((s.centroids-original_centroids[name])**2)
                                for name,s in sources.items()]).mean()
            loss=objective+POLICY.fp32_centroid_guard*anchor
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite all-layer recovery loss")
            loss.backward()
        finally:
            hook.remove()
        if len(captured)!=1 or captured[0].grad is None:
            raise AssertionError("No through-model TRAIN gradient for original operator "+target_name)
        gradient=captured[0].grad.detach().cpu().numpy().copy()
        active=sum(x.centroids.grad is not None for x in sources.values())
        if active!=P.CODEBOOK_MATRIX_COUNT:
            raise AssertionError("Green Tunnel did NOT train all model levels: "+str(active))
        norm=float(torch.nn.utils.clip_grad_norm_(
            [x.centroids for x in sources.values()],POLICY.gradient_clip,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():s.centroids.clamp_(-16,16)
        clear(sources)
        # Evaluate an actual BEFORE/AFTER 5-bit code-index choice on
        # EXACT same TRAIN loss. Never access heldout VALID or public DEV.
        with torch.inference_mode():
            loss_post,_,_,_=choose_objective(
                student,step,train_teacher_refs,train_pairs)
            loss_before_codes=float(loss_post)
        proposed=propose_codes(target,gradient,original_fp32_matrices[target_name])
        if proposed["proposed_groups"]>POLICY.max_5bit_swaps_per_step:
            raise AssertionError("Paid 5bit source code-swap budget exceeded")
        count=proposed["proposed_groups"]
        total_proposed+=count
        accepted=False;candidate_loss=loss_before_codes
        if count:
            with torch.no_grad():apply_codes(target,proposed,True)
            clear(sources)
            with torch.inference_mode():
                candidate,_,_,_=choose_objective(
                    student,step,train_teacher_refs,train_pairs)
                candidate_loss=float(candidate)
            accepted=math.isfinite(candidate_loss) and (
                candidate_loss+1e-7<loss_before_codes)
            if not accepted:
                with torch.no_grad():apply_codes(target,proposed,False)
                clear(sources)
            else:
                total_accepted+=count
        row={"cycle":step//POLICY.steps_per_cycle+1,
             "step":step+1,"domain":kind,
             "training_case_index":index,
             "original_operator_receiving_5bit_code_proposal":target_name,
             "all_211_packed_operators_received_GRAD":active,
             "TRAIN_full_vocab_or_balanced_objective":float(objective.detach()),
             "TRAIN_source_centroid_guard":float(anchor.detach()),
             "TRAIN_code_swap_before":loss_before_codes,
             "TRAIN_code_swap_after":candidate_loss,
             "original_gradient_norm_before_clipping":norm,
             "candidate_original_5bit_group_swaps":count,
             "accepted_5bit_group_swaps":count if accepted else 0,
             "accepted_ONLY_for_improvement_on_same_TRAIN":accepted,
             "objective_metrics_TRAIN_ONLY_NOT_QA":stats}
        records.append(row)
        clear(sources)
        if (step+1)%12==0:
            print(json.dumps({"stage":"v196_green_tunnel_full_vocab_211_layers",
                "cycle":row["cycle"],"step":row["step"],
                "source_operator":target_name,"source_code_accepted":row["accepted_5bit_group_swaps"],
                "all_codebooks_gradient_count":active}),flush=True)
        if (step+1)%24==0:
            save(out/"TRAIN_incremental_green_tunnel_loss_only.json",{
                "steps_completed":step+1,"last_three":records[-3:],
                "total_candidate_5bit_swaps":total_proposed,
                "total_accepted_5bit_swaps":total_accepted,
                "QA_or_validation_opened":False,
                "teacher_full_vocab_token_count":P.VOCAB_SIZE})
    if len(records)!=total_steps:
        raise AssertionError("Incomplete whole-vocabulary/cycle restoration")
    code_net=sum(int(torch.count_nonzero(s.codes-original_codes[name]))
                 for name,s in sources.items())
    changed_centroids=sum(int(torch.count_nonzero(
        s.centroids.detach().to(torch.float16)-
        original_centroids[name].to(torch.float16)))
        for name,s in sources.items())
    if changed_centroids==0:
        raise AssertionError("No physically stored paid centroids were changed")
    return {"full_original_embedding_attention_MLP_codebooks_trained_each_step":True,
        "source_trainable_packed_matrix_count":len(sources),
        "output_vocabulary_size_all_words_proba_gradient":P.VOCAB_SIZE,
        "beginning_from_physically_trained_v186_135M":True,
        "completed_recovery_cycles":POLICY.cycles,
        "scheduled_full_model_gradient_steps":total_steps,
        "all_layers_gradient_on_every_step":True,
        "full_vocab_training_only_and_contradictory_PUBLIC_TRAIN":True,
        "total_candidate_5bit_code_swaps":total_proposed,
        "total_accepted_5bit_code_swaps":total_accepted,
        "actual_net_5bit_code_changes":code_net,
        "FP16_paid_centroid_cells_net_changed":changed_centroids,
        "all source updates stay in already paid 5bit codes and FP16 codebooks":True,
        "no_unpaid_runtime_GreenTunnel_neural_parameters":True,
        "no_dev_or_valid_used_for_step_acceptance":True,
        "no_KL_NLL_PPL_or_TRAIN_loss_as_quality_gate":True,
        "complete_TRAIN_only_steps":records}
