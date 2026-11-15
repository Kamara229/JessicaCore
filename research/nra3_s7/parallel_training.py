"""S7: train genuine parallel relational reasoning inside unified compressed NRA3.

No new matrices: reuse existing S6 6,106,154-parameter CB59 source.
UD syntax gold is admitted ONLY during TRAIN; inference consumes the
NRA's own role logits and causal hidden states. Relational prediction must
learn head links even if dependent precedes head in a completed prefix.

No independent evaluation, original CALIBRI weights, Green Tunnel updates,
or candidate answer route. TRAIN-only result cannot prove generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s4.complete_prefix_training_contract import (
    CausalAlignment,train_only_multitask_loss,
)
from research.nra3_s4.russian_ud_train_only import (
    RussianTrainSentence,TRAIN_SPLIT,
)
from research.nra3_s6.physical_joint_grammar_qat import PhysicallyPaidNRA3QAT


@dataclass
class S7ParallelObjective:
    total:torch.Tensor
    grammar_semantic:torch.Tensor
    relational_ud_head:torch.Tensor
    mean_hypothesis_entropy:torch.Tensor
    branch_means:torch.Tensor


def parallel_train_objective(area:NRA3NeuralArea,
                             hidden320:torch.Tensor,
                             row:RussianTrainSentence,
                             aligned:CausalAlignment)->S7ParallelObjective:
    """Full-prefix UD head loss supervised on the SAME 12 locked TRAIN rows."""
    if (not isinstance(area,NRA3NeuralArea) or
        area.config.parallel_hypotheses!=3 or row.split!=TRAIN_SPLIT or
        aligned.independent_eval_labels_loaded or
        aligned.original_prompt_sha256!=row.sha256() or
        hidden320.ndim!=3 or hidden320.shape!=(1,aligned.tokenizer_token_count,320)):
        raise ValueError("S7 only admits original complete-prefix Russian TRAIN")
    features=area(hidden320)
    if (features.parallel_hypothesis_gates is None or
        features.parallel_hypothesis_gates.shape!=(1,hidden320.shape[1],3)):
        raise ValueError("S7 actual parallel hypothesis evidence missing")
    grammar=train_only_multitask_loss(
        area,hidden320,row,aligned,feature_override=features)
    ids=torch.tensor([w.last_token for w in aligned.word_alignment],
                     device=hidden320.device,dtype=torch.long)
    rel=features.relations[0,ids,:]            # [W,R] past-only states
    root=features.relations[0,aligned.complete_prefix_position,:]  # full visible prefix
    query=area.rel_q(rel)
    keys=area.rel_k(rel)
    root_key=area.rel_k(root)
    logits=torch.cat((query@keys.T,
                      (query@root_key[:,None])),dim=-1)
    logits=logits/math.sqrt(area.config.relation_width)
    # All heads in the fully visible *current* prefix are admissible; no
    # future-generation token or externally supplied graph enters NRA forward.
    w=len(row.words)
    labels=torch.tensor([node.head-1 if node.head!=0 else w
                         for node in row.words],
                        dtype=torch.long,device=hidden320.device)
    loss_relation=F.cross_entropy(logits,labels)
    gates=features.parallel_hypothesis_gates[0,ids,:]
    entropy=-(gates*gates.clamp_min(1e-9).log()).sum(-1).mean()
    # Make hypothesis selector trainable through PAID original rel_q/k/v and
    # role_head; tiny entropy regularizer penalizes trivial branch collapse.
    total=grammar.total+0.35*loss_relation-0.01*entropy
    if (not bool(torch.isfinite(total)) or
        not bool(torch.isfinite(entropy)) or
        not bool(torch.isfinite(gates).all())):
        raise ValueError("S7 nonfinite parallel relational TRAIN loss")
    return S7ParallelObjective(total,grammar.total,loss_relation,entropy,
                               gates.mean(0))


class S7UnifiedParallelSupervision(nn.Module):
    """Same single in-area 43-paid-operator graph; changed objective ONLY."""
    def __init__(self,config:NRA3Config):
        super().__init__()
        if config.parallel_hypotheses!=3:
            raise ValueError("S7 cannot run without precisely three hypotheses")
        self.area=NRA3NeuralArea(config)

    def forward(self,hidden320,row,aligned):
        return parallel_train_objective(self.area,hidden320,row,aligned)


def attach_parallel_objective(qat:PhysicallyPaidNRA3QAT):
    """Replaces TRAIN loss wrapper, NOT the trained source or weight layout."""
    if (not isinstance(qat,PhysicallyPaidNRA3QAT) or
        qat.config.parallel_hypotheses!=3 or
        qat.output_gate!=0. or qat.candidate_accepted):
        raise ValueError("S7 requires fully frozen research QAT with 3 hypotheses")
    s7=S7UnifiedParallelSupervision(qat.config).cpu().eval()
    # Transfer exact reconstructed original PAID weight tensors. This is
    # temporary RAM for differentiable source-only TRAIN; no FP32 export.
    s7.area.load_state_dict(qat._host.area.state_dict(),strict=True)
    s7.requires_grad_(False)
    qat._host=s7
    if len(qat.names)!=43 or qat._host.area.parameter_count()!=6_106_154:
        raise AssertionError("S7 changed original physical source count")
    return qat
