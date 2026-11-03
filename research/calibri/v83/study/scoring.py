"""Pure TRAIN-only triage. Error magnitude is NOT causal token sensitivity."""
from __future__ import annotations
import math
from v83.measure.metrics import Moment


def summarize(book) -> dict:
    snap = book.snapshots()
    local = snap["teacher_forced_local_projection"]
    propagated = snap["propagated_layer_output"]
    if not local or not propagated:
        raise AssertionError("both local and propagated audit required")
    all_error = sum(x["squared_error_energy"] for x in local.values())
    if not all_error > 0:
        raise AssertionError("trained CB59 source cannot be identical to FP32")
    ranked = sorted(local.items(), key=lambda pair: pair[1]["squared_error_energy"],
                    reverse=True)
    per_layer = {}
    for path, data in local.items():
        layer_number = int(path.split(".")[2])
        key = f"layer_{layer_number:02d}"
        row = per_layer.setdefault(key, {
            "squared_error_energy": 0.0,
            "teacher_squared_energy": 0.0, "projected_operators": 0})
        row["squared_error_energy"] += data["squared_error_energy"]
        row["teacher_squared_energy"] += data["teacher_squared_energy"]
        row["projected_operators"] += 1
    for row in per_layer.values():
        row["aggregate_relative_L2"] = math.sqrt(
            row["squared_error_energy"] / row["teacher_squared_energy"])
        row["fraction_of_all_projection_error_energy"] = (
            row["squared_error_energy"] / all_error)
    return {
        "measured_operators": len(local),
        "measured_layers": len(propagated),
        "highest_local_teacher_forced_error_energy": [
            {"path": p, "error_energy_share": x["squared_error_energy"] / all_error,
             "relative_L2": x["relative_L2"],
             "observations": x["observations"]} for p, x in ranked[:12]
        ],
        "local_projection_metrics": local,
        "propagated_layer_metrics": propagated,
        "layer_aggregates": per_layer,
        "diagnostic_not_causal_bit_allocation": True,
        "no_proposal_to_change_existing_model_or_bits_based_on_unseen_TEST": True,
        "interpretation": (
            "Direct local errors use teacher activations and isolate CB59 "
            "operator approximation. Propagated errors measure cumulative drift. "
            "Magnitude or error-energy ranking alone does not establish task "
            "importance, optimize allocation, or demonstrate quality recovery."
        ),
    }
