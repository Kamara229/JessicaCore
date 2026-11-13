"""NRA 2.0 standalone CPU TRAIN + exploratory unseen-chain-length checks.

No CALIBRI checkpoint, Green Tunnel, J7, 456/128/32, QA or held-out TEST is
opened. This is a narrow synthetic relational task, NOT language understanding.
Validation is an exploratory development set; fresh, sealed evaluation is
necessary before asserting model-wide generalization.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn

from .model import RelationalNRA

TRAIN_NODES = 9
TRAIN_DISTANCES = (1, 2, 3, 4, 5)
DEV_NODES = 10
DEV_DISTANCES = (5,)  # new graph size and unseen permutations; depth seen in TRAIN
MODEL_SEED = 19522026


def graph_batch(seed: int, nodes: int, size: int, distances: tuple[int, ...]):
    """Fresh random permutations; balanced positive and reverse-direction cases."""
    if nodes < 3 or size < 2 or size % 2 or not distances:
        raise ValueError("Invalid graph batch shape")
    if any(d < 1 or d > nodes - 3 for d in distances):
        raise ValueError("Distance outside graph")
    rng = random.Random(seed)
    adj = torch.zeros(size, nodes, nodes)
    sources = torch.zeros(size, nodes)
    targets = torch.zeros(size, nodes)
    labels = torch.zeros(size)
    for item in range(size):
        order = list(range(nodes))
        rng.shuffle(order)
        for left, right in zip(order[:-1], order[1:]):
            adj[item, left, right] = 1.0
        distance = rng.choice(distances)
        start = rng.randrange(1, nodes - distance - 1)  # keep both endpoints internal
        left, right = order[start], order[start + distance]
        yes = item % 2 == 0
        if not yes:
            left, right = right, left
        sources[item, left] = 1.0
        targets[item, right] = 1.0
        labels[item] = float(yes)
    return adj, sources, targets, labels


@torch.no_grad()
def evaluate(model: RelationalNRA, *, seed: int, nodes: int,
             distances: tuple[int, ...], size: int = 1024) -> dict:
    model.eval()
    edges, source, target, labels = graph_batch(seed, nodes, size, distances)
    logits = model(edges, source, target)
    predicted = (logits > 0).float()
    correct = (predicted == labels)
    yes = labels == 1
    no = ~yes
    return {
        "correct": int(correct.sum()), "total": size,
        "accuracy": float(correct.float().mean()),
        "positive_accuracy": float(correct[yes].float().mean()),
        "negative_accuracy": float(correct[no].float().mean()),
        "nodes": nodes, "distances": list(distances), "seed": seed,
    }


def run(*, out: Path, steps: int = 300, learning_rate: float = .006,
        train_size: int = 96) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Immutable standalone NRA output directory required")
    if steps < 1 or learning_rate <= 0 or train_size % 2 or train_size < 2:
        raise ValueError("Invalid TRAIN hyperparameters")
    torch.set_num_threads(1)
    torch.manual_seed(MODEL_SEED)
    model = RelationalNRA(width=16, hops=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    first_loss = None
    for step in range(1, steps + 1):
        model.train()
        edges, source, target, labels = graph_batch(
            100000 + step, TRAIN_NODES, train_size, TRAIN_DISTANCES)
        optimizer.zero_grad(set_to_none=True)
        logits = model(edges, source, target)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters()):
            raise AssertionError("NRA gradients missing or invalid")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # The exploratory dev set is never read during optimizer updates.
    trained = evaluate(model, seed=310001, nodes=TRAIN_NODES,
                       distances=TRAIN_DISTANCES)
    unseen = evaluate(model, seed=320001, nodes=DEV_NODES,
                      distances=DEV_DISTANCES)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "NRA2_STANDALONE_RELATIONAL_FP16.pt"
    physical = {key: value.detach().cpu().to(torch.float16).contiguous()
                for key, value in model.state_dict().items()}
    torch.save(physical, checkpoint)
    restored = RelationalNRA(width=16, hops=5)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                        weights_only=True), strict=True)
    for key, value in restored.state_dict().items():
        if not torch.equal(value.half(), physical[key]):
            raise AssertionError("Physical FP16 NRA tensor mismatch after reload")
    reread = evaluate(restored, seed=320001, nodes=DEV_NODES,
                      distances=DEV_DISTANCES)
    if reread["correct"] != unseen["correct"]:
        raise AssertionError("Physical FP16 NRA reload changed OOD correctness")
    if not (trained["accuracy"] >= .98 and unseen["accuracy"] >= .95):
        qualification = "NOT_QUALIFIED"
    else:
        qualification = "EXPLORATORY_STANDALONE_PASS_NOT_INTEGRATION"
    report = {
        "stage": "NRA2_STANDALONE_RELATIONAL_V0",
        "candidate_status": qualification,
        "scope": "synthetic directed-chain transitivity only",
        "not_a_language_generalization_claim": True,
        "development_validation_not_sealed_QA": True,
        "training": {"steps": steps, "samples_per_step": train_size,
                     "learning_rate": learning_rate, "first_loss": first_loss,
                     "last_loss": float(loss.detach())},
        "architecture": {"width": model.width, "hops": model.hops,
                         "trainable_parameters": model.parameter_count()},
        "train_distribution": trained,
        "novel_graph_size_five_hop_dev": unseen,
        "extra_six_hop_stress_not_gate": evaluate(
            restored, seed=330001, nodes=11, distances=(6,)),
        "fp16_reloaded_dev": reread,
        "fp16_checkpoint_bytes": checkpoint.stat().st_size,
        "fp16_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "calibri_2_integrated": False,
        "no_base_model_or_existing_NRA_modified": True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes": True,
    }
    (out / "NRA2_STANDALONE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    run(out=args.out, steps=args.steps)


if __name__ == "__main__":
    main()
