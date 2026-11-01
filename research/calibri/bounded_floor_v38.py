"""CALIBRI v38 — physical equal-byte bounded inter-layer bit allocation.

Protect EVERY one of 12 real Pythia MLP matrices with a TRAIN-only
minimum 75% / maximum 125% of its v36 locally allotted INT4 ROWS.
This responds to v37's measured starvation of six down-projections.
All row choices and module sensitivity come from WikiText TRAIN only.
The total physically serialized MLP bundle must be exactly 343840 bytes.
"""
from __future__ import annotations
import math
import json
import numpy as np
import torch
from direct_tiles_v33 import parse
from interlayer_alloc_v37 import (
    module_identity, pack_row_assignment, true_train_row_errors,
    package, GLOBAL_LIMIT_BYTES,
)
from quality import ppl_eval

FLOOR_FRACTION=0.75
CEILING_FRACTION=1.25


def bounded_floor_allocator(names,matrices,inputs,source_train_nll,model,local_files,train_tokens,context):
    if set(local_files)!=set(names) or set(matrices)!=set(names) or set(inputs)!=set(names):
        raise AssertionError("all 12 original matrices, training captures and controls required")
    all3={}
    scores={}
    penalties={}
    boundaries={}
    baseline_4bit_by_module={}
    for n in names:
        w=matrices[n]
        x=inputs[n]
        rows,width=w.shape
        if width%8:
            raise AssertionError("row cost must be integer packed bytes")
        _m,_nn,br,bc,tiles=parse(local_files[n])
        if (br!=1 or bc!=width or _m!=rows or _nn!=width or len(tiles)!=rows):
            raise AssertionError("v36 control does not contain every input row")
        local_4=sum(t[2]==4 for t in tiles)
        if not (0<local_4<rows):
            raise AssertionError("nontrivial original local allocation required")
        floor=math.ceil(local_4*FLOOR_FRACTION)
        ceiling=min(rows,math.floor(local_4*CEILING_FRACTION))
        if not (0<floor<=local_4<=ceiling<rows):
            raise AssertionError("preregistered min/max protection bounds invalid")
        boundaries[n]={"local_4bit_rows":local_4,
                       "minimum_4bit_rows":floor,
                       "maximum_4bit_rows":ceiling,
                       "cost_per_upgraded_row_bytes":width//8,
                       "row_count":rows}
        baseline_4bit_by_module[n]=local_4
        s3,s4,q3=true_train_row_errors(w,x)
        with torch.inference_mode():
            target=model.get_submodule(n)
            source=target.weight.detach().clone()
            try:
                target.weight.copy_(torch.from_numpy(q3))
                perturbed_NLL=ppl_eval(model,train_tokens,context)["nll"]
            finally:
                target.weight.copy_(source)
        penalties[n]=max(float(perturbed_NLL-source_train_nll),1e-8)
        scores[n]=np.maximum(0.0,s3-s4)
        all3[n]=[3]*rows
        print(json.dumps({
            "training_only_module":n,
            "train_source_NLL":source_train_nll,
            "train_pure_INT3_NLL":perturbed_NLL,
            "relative_NLL_penalty":penalties[n],
            "predeclared_4bit_rows_floor":floor,
            "predeclared_4bit_rows_ceiling":ceiling,
        }),flush=True)

    median=float(np.median(list(penalties.values())))
    multipliers={n:float(np.clip(penalties[n]/max(median,1e-8),0.25,4.0))
                 for n in names}
    all3_files={n:pack_row_assignment(matrices[n],all3[n]) for n in names}
    base_size=len(package(all3_files,1))
    remaining=GLOBAL_LIMIT_BYTES-base_size
    if remaining<=0:
        raise AssertionError("invalid global size control versus uniform INT3")
    chosen={}
    total_upgrades=0
    # Phase 1: protect every matrix's numerical precision, picking its
    # best TRAIN-output-SSE rows before cross-layer reallocation.
    for n in names:
        floor=boundaries[n]["minimum_4bit_rows"]
        cost=boundaries[n]["cost_per_upgraded_row_bytes"]
        ranks=sorted(range(matrices[n].shape[0]),key=lambda i:(-float(scores[n][i]),i))
        selected=set(ranks[:floor])
        if remaining<floor*cost:
            raise AssertionError("per-module floors do not fit the v36 physical byte cap")
        for i in selected:
            all3[n][i]=4
        remaining-=floor*cost
        chosen[n]=selected
        total_upgrades+=floor

    # Phase 2: deterministic greedy TRAIN-only incremental benefit/byte,
    # but each module has a strict precommitted 125% local precision ceiling.
    options=[]
    for n in names:
        cost=boundaries[n]["cost_per_upgraded_row_bytes"]
        for row,gain in enumerate(scores[n]):
            if row in chosen[n] or gain<=0.:
                continue
            score=(float(gain)*multipliers[n])/float(cost)
            options.append((-score,module_identity(n),row,n,cost))
    options.sort()
    for _negative_score,_ident,row,n,cost in options:
        if remaining<cost or len(chosen[n])>=boundaries[n]["maximum_4bit_rows"]:
            continue
        chosen[n].add(row)
        all3[n][row]=4
        remaining-=cost
        total_upgrades+=1

    physical={n:pack_row_assignment(matrices[n],all3[n]) for n in names}
    size=len(package(physical,1))
    if size!=GLOBAL_LIMIT_BYTES or remaining!=0:
        raise AssertionError(f"not exactly fair v36 physical budget: used={size}, free={remaining}")
    for n in names:
        floor=boundaries[n]["minimum_4bit_rows"]
        top=boundaries[n]["maximum_4bit_rows"]
        actual=len(chosen[n])
        if not (floor<=actual<=top):
            raise AssertionError("per-matrix protection lost on "+n)
        boundaries[n]["final_4bit_rows"]=actual
        boundaries[n]["change_vs_local_4bit_rows"]=actual-baseline_4bit_by_module[n]
    return physical,{
        "method":"TRAIN-only per-module 75% floor/125% ceiling of local INT4 row counts; TRAIN single-module INT3 full-model NLL sensitivity times row actual output-SSE benefit per incremental byte, greedy within bounds",
        "floor_fraction":FLOOR_FRACTION,
        "ceiling_fraction":CEILING_FRACTION,
        "base_pure_INT3_bundle_bytes":base_size,
        "global_fixed_max_bytes":GLOBAL_LIMIT_BYTES,
        "greedy_used_bytes":size,
        "unused_global_budget_bytes":remaining,
        "total_rows_upgraded":total_upgrades,
        "train_only_pure_INT3_layer_NLL_penalties":penalties,
        "train_only_clipped_layer_multipliers":multipliers,
        "bounded_precisions_by_module":boundaries,
        "row_upgrades_to_INT4_per_module":{n:len(chosen[n]) for n in names},
        "train_only_selection_not_checked_against_validation":True,
    }
