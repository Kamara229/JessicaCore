"""NRA2 S3CE: order-paired semantic-role TRAIN-only objective, no model inference.

S3CD consumed fifth independent128 and diagnosed a 13/64 destination-first
regression. Its scores, questions, labels and CPU tensors MUST NOT be used here.
S3BR TRAIN-only has exact same-direction meaning-preserving even/odd style
pairs; S3BX permits only fit192 (96 pairs). The other 320 S3BX TRAIN-derived
guards are not fresh independent samples, and must not enter fit/selection.

No S3BZ/S3BT initialization; future learner must start from frozen PRE-S3BR
S3BN (physical FP16 SHA). This is a reusable objective with synthetic tests,
NOT trained new model, not a claim of generalization and not model promotion.
LIVE, main, QA/TEST, Render, Supabase, original43M, Green Tunnel, NRA, 616 J7
unchanged. Output gate 0, candidate_accepted false.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import (
    split_data, protocol, PRE_S3BR_PARENT_SHA,
)
from research.nra2_bridge.stage3br_relation_language_train512_prereg import (
    SPLIT as S3BR_TRAIN, DESTINATION_FIRST,
)

STAGE = "NRA2_S3CE_PAIRED_SEMANTIC_ORDER_FIT192_OBJECTIVE_NO_TRAINING"
S3BZ_SHA = "6084061cb1f9ea46ef3b504179bee09b392f4427ac52ec58d768645da98872ee"
FIFTH_EVAL_CONSUMED_IN = 43677179638
FIFTH_EXTERNAL_RESULTS_FORBIDDEN = True
PAIR_COUNT = 96
FIT_COUNT = 192
NODE_COUNT = 6
SYMMETRIC_KL_WEIGHT = 0.5
DIRECTION_GAP_SQUARED_WEIGHT = 0.1
OLD_TEACHER_KL_WEIGHT = 1.0
TEACHER_TEMPERATURE = 2.0


def require(ok, reason):
    if not bool(ok):
        raise AssertionError("S3CE FAIL CLOSED: " + reason)


@dataclass(frozen=True)
class FitOrderPairs:
    fit_prompt_sha256: tuple[str, ...]
    # Each pair references the same graph, semantic direction and construction.
    # Index zero: source entity first in TEXT; index one: destination first.
    pairs: tuple[tuple[int, int], ...]
    src_ids: tuple[int, ...]
    dst_ids: tuple[int, ...]
    yes_labels: tuple[int, ...]
    protocol_sha256: str

    def torch_pair_indices(self):
        return torch.tensor(self.pairs, dtype=torch.long)

    def gold(self):
        return (torch.tensor(self.src_ids, dtype=torch.long),
                torch.tensor(self.dst_ids, dtype=torch.long))


def build_fit_pairs() -> FitOrderPairs:
    groups = split_data()
    precommitted = protocol(groups)
    fit = groups["fit"]
    require(len(fit) == FIT_COUNT and all(r.split == S3BR_TRAIN for r in fit),
            "fit192 only; no grammar/topology/double guard rows")
    fit_h = {r.prompt_sha256 for r in fit}
    guard_h = {r.prompt_sha256 for k in ("grammar_guard", "topology_guard",
                                       "double_guard") for r in groups[k]}
    require(not fit_h & guard_h and len(guard_h) == 320,
            "guard320 leakage into fit")
    lookup = {}
    for i, r in enumerate(fit):
        require(r.query_style in (0, 1, 2, 3)
                and 0 <= r.src < NODE_COUNT and 0 <= r.dst < NODE_COUNT
                and r.src != r.dst,
                "only S3BX two fit grammar families/semantic roles permitted")
        key = (r.graph_id, r.query_style // 2)
        lookup.setdefault(key, []).append((i, r))
    require(len(lookup) == PAIR_COUNT and all(len(x) == 2 for x in lookup.values()),
            "96 matched graph+grammar pairs required")
    pairs = []
    for key in sorted(lookup):
        items = sorted(lookup[key], key=lambda v: v[1].query_style)
        (a, src_first), (b, dst_first) = items
        require(src_first.query_style % 2 == 0
                and dst_first.query_style % 2 == 1
                and dst_first.query_style in DESTINATION_FIRST
                and src_first.query_style not in DESTINATION_FIRST
                and src_first.graph_id == dst_first.graph_id
                and src_first.topology_key == dst_first.topology_key
                and src_first.edges == dst_first.edges
                and src_first.family == dst_first.family
                and src_first.src == dst_first.src
                and src_first.dst == dst_first.dst
                and src_first.reachable == dst_first.reachable
                and src_first.prompt_sha256 != dst_first.prompt_sha256,
                "paraphrase must preserve semantic source and destination")
        pairs.append((a, b))
    seen = sorted(i for pair in pairs for i in pair)
    require(seen == list(range(FIT_COUNT)), "every allowed fit row exactly once")
    require(len({fit[a].topology_key for a,b in pairs}) == 48,
            "source fit192 graph/topology ABI changed")
    require(sum(bool(fit[a].reachable) for a,b in pairs) == 48,
            "yes/no matched semantic pair skew")
    return FitOrderPairs(
        tuple(r.prompt_sha256 for r in fit),tuple(pairs),
        tuple(r.src for r in fit), tuple(r.dst for r in fit),
        tuple(int(r.reachable) for r in fit),
        precommitted["commitment_sha256"],
    )


def _sym_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(a.float(), dim=-1)
    log_b = F.log_softmax(b.float(), dim=-1)
    pa = log_a.exp()
    pb = log_b.exp()
    return 0.5*((pa*(log_a-log_b)).sum(-1)
                + (pb*(log_b-log_a)).sum(-1)).mean()


def objective(
    src_logits: torch.Tensor,
    dst_logits: torch.Tensor,
    layout: FitOrderPairs,
    teacher_logits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable TRAIN-only supervised + orientation consistency loss.

    Train-time q.node_order must be computed from original text offsets, NOT
    from gold labels. This objective receives gold only for CE training.
    It can make source-first/destination-first paraphrases agree but must NOT
    force opposite semantic directions (src->dst vs dst->src) to agree.
    """
    n = FIT_COUNT
    require(src_logits.shape == (n, NODE_COUNT)
            and dst_logits.shape == (n, NODE_COUNT)
            and torch.is_floating_point(src_logits)
            and torch.is_floating_point(dst_logits)
            and bool(torch.isfinite(src_logits).all())
            and bool(torch.isfinite(dst_logits).all()),
            "fit192 role logits must be finite [192,6] from TRAIN fit only")
    pairs = layout.torch_pair_indices().to(src_logits.device)
    require(pairs.shape == (PAIR_COUNT, 2)
            and sorted(pairs.reshape(-1).tolist()) == list(range(n)),
            "invalid pair alignment")
    gold_src, gold_dst = layout.gold()
    gold_src = gold_src.to(src_logits.device)
    gold_dst = gold_dst.to(dst_logits.device)
    require(bool(torch.equal(gold_src[pairs[:,0]], gold_src[pairs[:,1]]))
            and bool(torch.equal(gold_dst[pairs[:,0]], gold_dst[pairs[:,1]])),
            "gold semantic roles must remain unchanged inside paired wording")
    src_ce = F.cross_entropy(src_logits.float(), gold_src, reduction="none")
    dst_ce = F.cross_entropy(dst_logits.float(), gold_dst, reduction="none")
    per_example = 0.5*(src_ce + dst_ce)
    first_ce = per_example[pairs[:,0]].mean()
    last_ce = per_example[pairs[:,1]].mean()
    # Equal first/last empirical risk without allowing one orientation to
    # dominate, plus a positive, differentiable balance penalty. Both should
    # decrease due to shared supervised CE, not merely converge upward.
    supervised = 0.5*(first_ce + last_ce)
    pair_kl = 0.5*(_sym_kl(src_logits[pairs[:,0]], src_logits[pairs[:,1]])
                   + _sym_kl(dst_logits[pairs[:,0]], dst_logits[pairs[:,1]]))
    disparity = (first_ce-last_ce).square()
    teacher_kl = torch.zeros((), device=src_logits.device, dtype=torch.float32)
    if teacher_logits is not None:
        ts,td = teacher_logits
        require(ts.shape == (n,NODE_COUNT) and td.shape == (n,NODE_COUNT)
                and bool(torch.isfinite(ts).all()) and bool(torch.isfinite(td).all()),
                "frozen S3BN teacher may only provide same fit192 logits")
        for student, teacher in ((src_logits, ts), (dst_logits, td)):
            log_s = F.log_softmax(student.float()/TEACHER_TEMPERATURE,dim=-1)
            probs_t = F.softmax(teacher.detach().float()/TEACHER_TEMPERATURE,dim=-1)
            teacher_kl = teacher_kl + 0.5*F.kl_div(
                log_s,probs_t,reduction="batchmean"
            )*(TEACHER_TEMPERATURE**2)
    total = (supervised + SYMMETRIC_KL_WEIGHT*pair_kl
             + DIRECTION_GAP_SQUARED_WEIGHT*disparity
             + OLD_TEACHER_KL_WEIGHT*teacher_kl)
    require(bool(torch.isfinite(total)), "nonfinite role-balance TRAIN loss")
    return {
        "total":total,"supervised":supervised,"source_first_ce":first_ce,
        "destination_first_ce":last_ce,"semantic_pair_symmetric_kl":pair_kl,
        "direction_ce_gap_squared":disparity,"frozen_S3BN_teacher_kl":teacher_kl,
    }


def protocol_report(layout: FitOrderPairs):
    obj = {
        "stage":STAGE,
        "only_S3BX_fit192_pairs":True,"fit_count":FIT_COUNT,
        "same_semantics_sourcefirst_destinationfirst_pairs":PAIR_COUNT,
        "old_TRAINS3BN_safe_init_sha256":PRE_S3BR_PARENT_SHA,
        "rejected_S3BZ_not_initialized_or_scored_sha256":S3BZ_SHA,
        "source_S3BX_blocked_commitment_sha256":layout.protocol_sha256,
        "fit192_prompt_hashes_sha256":hashlib.sha256(
            json.dumps(layout.fit_prompt_sha256,separators=(",",":")).encode()
        ).hexdigest(),
        "fit_pair_indices_sha256":hashlib.sha256(
            json.dumps(layout.pairs,separators=(",",":")).encode()
        ).hexdigest(),
        "fit_yes_count":sum(layout.yes_labels),
        "fifth_independent_EVAL128_permanently_consumed_run":FIFTH_EVAL_CONSUMED_IN,
        "all_five_independent_EVAL128_excluded_from_gradient_scoring_selection":True,
        "guard320_excluded_from_gradient_and_selection":True,
        "no_decoder_cpu_weights_model_inference_or_training_this_step":True,
        "future_loss_terms":["balanced_groundtruth_role_CE",
                             "source_destination_first_symmetric_KL",
                             "source_destination_ce_gap_squared",
                             "S3BN_teacher_KL_optional"],
        "independent_EVAL_transfer_not_established":True,
        "candidate_accepted":False,"output_gate":0.0,
        "LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    obj["commitment_sha256"] = hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()
    return obj


def main():
    require(len(sys.argv)==2, "single NEW output directory required")
    path=Path(sys.argv[1])
    require(not path.exists(),"must not overwrite immutable research results")
    layout=build_fit_pairs()
    report=protocol_report(layout)
    path.mkdir(parents=True,exist_ok=False)
    (path/"NRA2_S3CE_FIT192_ORDER_PAIR_OBJECTIVE_PROTOCOL.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({"stage":STAGE,"fit":FIT_COUNT,"pairs":PAIR_COUNT,
                      "commitment_sha256":report["commitment_sha256"],
                      "output_gate":0.0,"candidate_accepted":False},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
