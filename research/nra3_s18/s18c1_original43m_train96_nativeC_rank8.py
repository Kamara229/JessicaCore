"""S18c1: physically train ONE paid 2,309-FP16 rank8 directed Russian role head.

Data: S18b frozen original 43M+Green Tunnel real CPU layer8 NEW96 TRAIN only.
Feature: all 43 original CB59 packed C matrices and frozen S11 tiny 39-FP16
router executed with true native C, original token IDs/word boundaries.
Gold: all 300 directed argument roles across 96 new TRAIN sentences.
No oracle filtering of candidate pairs; both directions for every two words.
No external EVAL, no selection by any heldout, no model promotion or LIVE.
S95 may condition on full text: DO NOT claim strict online causal semantics.
Only new readout head trains; physically bill all FP16 weights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.real43m_train72_readouts import (
    S8B_43_CB59_SHA, S11_39_FP16_SHA,
)
from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence, PARAMETERS, PHYSICAL_BYTES, ROLES,
)
from research.nra3_s16.real43m_train72_directed_valence import (
    DirectedPairTrain, frozen_paired_logits, metrics, score_weighted_train,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256, direct_predicate_argument_gold,
)
from research.nra3_s18.real43m_train96_physical_source import (
    checked_real_train96, grouped_train96_split_contract, S18B_TRUE_CPU_RUN,
)

STEPS = 200
CHECKPOINT_STEPS = (0, 50, 100, 150, 200)
LR = 0.025
WEIGHTS = torch.tensor([.20, 1., 1., 1., 1.5], dtype=torch.float32)
TOTAL_PAIRS = 3288
TOTAL_ROLES = 300


def require(ok, why):
    if not bool(ok):
        raise ValueError("S18c1 frozen original43M+GT native-C true TRAIN96 " + why)


def native_pair_train96(samples, native):
    """Execute original native-C on EVERY original wordpiece, both directions.

    POS/UD/semantic gold only constructs labels AFTER source feature computation.
    Every distinct pair including negative pairs is scored both ways.
    """
    pvec, avec, ppos, apos, current, target, orientations, source = (
        [], [], [], [], [], [], [], [])
    relations_sha = []
    n_gold = 0
    with torch.inference_mode():
        for k, sample in enumerate(samples):
            row = sample.row
            h = sample.hidden320
            n = int(h.shape[0])
            require(h.shape == (n, 320) and bool(torch.isfinite(h).all()),
                    "frozen original43M+GT layer8 tensor invalid")
            out = native(h.unsqueeze(0))
            require(out.relations.shape == (1, n, 128)
                    and out.parallel_hypothesis_gates is not None
                    and out.parallel_hypothesis_gates.shape == (1, n, 3)
                    and bool(torch.isfinite(out.relations).all())
                    and native.output_gate == 0
                    and native.candidate_accepted is False,
                    "actual 43 original packed CB59 and frozen S11 native execution failed")
            rel = out.relations[0].detach().float().contiguous()
            positions = [int(w.last_token) for w in sample.aligned.word_alignment]
            require(len(positions) == len(row.words)
                    and all(0 <= j < n for j in positions)
                    and positions == sorted(set(positions)),
                    "wordpiece boundaries not strict actual original Calibri tokens")
            # Gold is NEVER supplied to native() or PaidDirectedValence inputs.
            gold = {(pred - 1, arg - 1): ROLES.index(role)
                    for pred, arg, role in direct_predicate_argument_gold(row)}
            seen = set()
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    for pred_idx, arg_idx in ((i, j), (j, i)):
                        a, b = positions[pred_idx], positions[arg_idx]
                        require(a != b and max(a, b) < n,
                                "invalid physical directed source/predicate token position")
                        y = gold.get((pred_idx, arg_idx), 0)
                        if y:
                            n_gold += 1
                            seen.add((pred_idx, arg_idx))
                        pvec.append(rel[a])
                        avec.append(rel[b])
                        ppos.append(a)
                        apos.append(b)
                        current.append(max(a, b))
                        target.append(y)
                        orientations.append(0 if pred_idx > arg_idx else 1)
                        source.append(k)
            require(seen == set(gold),
                    "full directed gold map not representable in unfiltered pairs")
            relations_sha.append(hashlib.sha256(
                rel.numpy().astype("<f4", copy=False).tobytes()).hexdigest())
    long = lambda values: torch.tensor(values, dtype=torch.long)
    x = DirectedPairTrain(
        torch.stack(pvec).contiguous(), torch.stack(avec).contiguous(),
        long(ppos), long(apos), long(current), long(target),
        long(orientations), long(source), len(samples), n_gold,
    )
    require(x.rows == 96 and x.positive_links == TOTAL_ROLES
            and x.total_pairs == TOTAL_PAIRS
            and x.predicate.shape == x.argument.shape == (TOTAL_PAIRS, 128)
            and set(x.orientation.tolist()) == {0, 1}
            and int((x.target != 0).sum()) == TOTAL_ROLES
            and all(int((x.target == cls).sum()) == expect for cls, expect in (
                (1, 96), (2, 96), (3, 96), (4, 12))),
            "real96 300 gold positive directed links or 3288 all-pair candidate mismatch")
    return x, relations_sha


def train_fixed_full96(x):
    """Single fixed 200-step optimizer, final FP16 checkpoint only.

    Snapshot training metrics are descriptive, NOT model selection criteria.
    """
    torch.manual_seed(318001)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = PaidDirectedValence().eval()
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS == 2309
            and model.output_gate == 0 and model.candidate_accepted is False,
            "paid rank8 model dimensions/output gate changed")
    snapshots = []

    def snapshot(step):
        blob = model.physical_bytes()
        paid = PaidDirectedValence.from_physical(blob).eval()
        require(len(blob) == PHYSICAL_BYTES == 4665
                and paid.physical_bytes() == blob
                and all(not p.requires_grad for p in paid.parameters()),
                "FP16 rank8 physical ABI failed exact byte-for-byte reload")
        with torch.inference_mode():
            m = metrics(frozen_paired_logits(paid, x), x)
        snapshots.append({
            "step": step, "physical_sha256": hashlib.sha256(blob).hexdigest(),
            "TRAIN_ONLY_full_directed_roles": m,
        })
        return blob

    snapshot(0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0002)
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss = score_weighted_train(model, x, WEIGHTS)
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "new paid rank8 TRAIN gradients missing or non-finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step in CHECKPOINT_STEPS[1:]:
            physical = snapshot(step)
    require([s["step"] for s in snapshots] == list(CHECKPOINT_STEPS)
            and len(physical) == PHYSICAL_BYTES,
            "only fixed final 200-step source accepted, no post-hoc train selection")
    return physical, snapshots


def main():
    parser = argparse.ArgumentParser()
    for name in ("s18b-hidden", "s18b-report", "s8-cb59", "s11-router", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    require(not a.out.exists(), "do not overwrite earlier physical source")
    samples, s18b = checked_real_train96(a.s18b_hidden, a.s18b_report)
    require(len(samples) == 96
            and s18b["all_616_native_CPU_generated_token_digests_unchanged"]
            and S18B_TRUE_CPU_RUN == 43910479658
            and S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256 ==
                "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9"
            and len(grouped_train96_split_contract()) == 12,
            "unverified original physical 43M CPU or new96 immutable source")
    original, router = a.s8_cb59.read_bytes(), a.s11_router.read_bytes()
    require(len(original) == 653798
            and hashlib.sha256(original).hexdigest() == S8B_43_CB59_SHA
            and len(CB59(original).handles) == 43
            and CB59(original).meta["original_unique_parameter_count"] == 6_106_154,
            "original packed CB59 43 native C weights fingerprint drift")
    require(len(router) == 123
            and hashlib.sha256(router).hexdigest() == S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes() == router,
            "original paid 39FP16 S11 fixed selector fingerprint drift")
    a.out.mkdir(parents=True, exist_ok=False)
    lib = a.out / "s18c1_original_43_CB59_native.so"
    subprocess.run(["cc", "-O3", "-std=c11", "-fPIC", "-shared",
        "research/calibri/direct_kernel_v59.c", "-o", str(lib)], check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native = NRA3NativePackedCPU(
        original, NRA3Config(parallel_hypotheses=3, parallel_selector_version=3),
        lib, paid_router_physical=router).eval()
    require(native.source_sha256 == S8B_43_CB59_SHA
            and native.paid_router.physical_bytes() == router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate == 0 and native.candidate_accepted is False
            and all(not p.requires_grad for p in native.parameters()),
            "original 43 native CB59 and S11 physically immutable")
    x, per_sentence_sha = native_pair_train96(samples, native)
    physical, snapshots = train_fixed_full96(x)
    require(len(physical) == PHYSICAL_BYTES
            and PaidDirectedValence.from_physical(physical).physical_bytes() == physical,
            "trained rank8 final FP16 physical reload mismatch")
    output = "NRA3_S18C1_TRUE43M_GT_NATIVE43_S11_NEW96_TRAIN300_RANK8_2309FP16.n3s16"
    (a.out / output).write_bytes(physical)
    require((a.out / output).read_bytes() == physical,
            "physically learned rank8 bytes corrupted")
    score = snapshots[-1]["TRAIN_ONLY_full_directed_roles"]
    summary = {
        "stage": "NRA3_S18C1_ORIGINAL43M_GT_NATIVE43_S11_NEW96_TRAIN300_PAID_RANK8",
        "S18b_original_43M_GT_true_96_CPU_run": S18B_TRUE_CPU_RUN,
        "S18b_original43M_physical_FP16_sha256": s18b["actual_fp16_feature_sha256"],
        "S18b_original_616_J7_generated_tokens_unchanged": True,
        "S18_source96_gold_SHA256": S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        "S8_original_43_packed_5bit_native_CB59_SHA256": S8B_43_CB59_SHA,
        "S11_original_39_FP16_router_SHA256": S11_39_FP16_SHA,
        "S18c1_real_original_native_C_on_96_hidden320": True,
        "S18c1_original_native_C_per_sentence_relation_SHA256": per_sentence_sha,
        "S18c1_training_rows_NEW96_only": 96,
        "S18c1_all_bidirectional_wordpair_candidates": x.total_pairs,
        "S18c1_positive_gold_roles": x.positive_links,
        "S18c1_gold_role_distribution": {ROLES[i]: int((x.target == i).sum()) for i in range(1, 5)},
        "S18c1_no_oracle_gold_or_POS_filter_in_candidates": True,
        "S18c1_no_label_entered_frozen_Calibri_or_native_C_or_S11": True,
        "S18c1_2309_rank8_FP16_physically_paid": True,
        "S18c1_2309_rank8_FP16_bytes": PHYSICAL_BYTES,
        "S18c1_rank8_paid_parameters": PARAMETERS,
        "S18c1_fixed_train_optimizer_steps": STEPS,
        "S18c1_fixed_snapshot_schedule": list(CHECKPOINT_STEPS),
        "S18c1_no_validation_based_selection": True,
        "S18c1_original43M_GT_native_C_S11_unchanged": True,
        "S18c1_model_file": output,
        "S18c1_model_sha256": hashlib.sha256(physical).hexdigest(),
        "S18c1_FINAL_ONLY_TRAIN_F1": score["all_two_direction_wordpair_exact_role"]["f1"],
        "S18c1_FINAL_ONLY_TRAIN_P": score["all_two_direction_wordpair_exact_role"]["precision"],
        "S18c1_FINAL_ONLY_TRAIN_R": score["all_two_direction_wordpair_exact_role"]["recall"],
        "S18c1_fixed_training_snapshots": snapshots,
        "S18c1_no_independent_eval_scored": True,
        "S18c1_consumed_S17c_S15c_S12c_EVAL_and_EVAL128_never_loaded": True,
        "S18c1_grouped_CV_accuracy_not_tested": True,
        "S18c1_original_S95_full_sentence_scope_causes_strict_online_causality_unproven": True,
        "S18c1_full_all_in_GT_NRA_bpp_le07_certified": False,
        "S18c1_CPU_speed_ge90pct_CB59_certified": False,
        "S18c1_candidate_active_616_J7_EOS_measured": False,
        "S18c1_output_gate": 0,
        "candidate_accepted": False,
        "activation_allowed": False,
        "no_LIVE_Jessica_chat_Render_Supabase_APK": True,
    }
    (a.out / "NRA3_S18C1_96_TRAIN_NATIVE_C_PAID_RANK8_REPORT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in
        ("S18c1_original_native_C_per_sentence_relation_SHA256",
         "S18c1_fixed_training_snapshots")}, ensure_ascii=False, sort_keys=True),
        flush=True)


if __name__ == "__main__":
    main()
