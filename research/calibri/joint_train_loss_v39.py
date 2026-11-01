"""CALIBRI v39: optimize ALL-12 calibrated model using actual joint TRAIN NLL.

This is a small, frozen, bounded and deterministic one- or four-row bit
swap search at exact SAME physically serialized 343,840 bytes. Real pretrained
Pythia source; no evaluation text enters proposals or their acceptance.
The model-level TRAIN NLL is measured with ALL 12 MLPs jointly compressed.
Only TRAIN activations are used for proposal ranking and scale calibration.
Negative/no-improvement outcomes are valid.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from calibrated_rows_v35 import calibrated_encode
from direct_tiles_v33 import parse,decode
from interlayer_alloc_v37 import pack_row_assignment,true_train_row_errors,GLOBAL_LIMIT_BYTES
from all_mlp_agnews_v36 import package,module_identity
from quality import ppl_eval

MAX_ROUNDS=2
MAX_PROBES_PER_ROUND=20
FLOOR_FRACTION=0.75
CEILING_FRACTION=1.25
MIN_ACCEPTED_NLL_REDUCTION=1e-7


def search_joint_train_nll(names,source_matrices,train_inputs,model,
                           original_modules,train_tokens,context,local_blobs):
    if set(names)!=set(source_matrices)!=set(train_inputs)!=set(original_modules)!=set(local_blobs) or len(names)!=12:
        raise AssertionError("v39 requires 12 matched original/real training source MLPs")
    states={}
    gain={}
    limits={}
    for n in names:
        m,w,br,bc,tiles=parse(local_blobs[n])
        original=source_matrices[n]
        if (br!=1 or m!=original.shape[0] or w!=original.shape[1]
            or bc!=w or len(tiles)!=m):
            raise AssertionError("original v36 physical per-row projection not valid")
        bits=np.array([row[2] for row in tiles],dtype=np.int8)
        local_count=int(np.count_nonzero(bits==4))
        low=math.ceil(local_count*FLOOR_FRACTION)
        high=min(m,math.floor(local_count*CEILING_FRACTION))
        if not (0<low<=local_count<=high<m):
            raise AssertionError("invalid predeclared precision constraints")
        s3,s4,_=true_train_row_errors(original,train_inputs[n])
        gain[n]=np.asarray(s3-s4,dtype=np.float64)
        states[n]=bits
        limits[n]={"local":local_count,"floor":low,"ceiling":high,"row_bytes":w//8}

    cached={}
    def serialize_candidate(n, bit_array):
        key=(n,bytes(bit_array.tobytes()))
        if key not in cached:
            encoded=pack_row_assignment(source_matrices[n],bit_array.tolist())
            recal,cal_info=calibrated_encode(source_matrices[n],train_inputs[n],encoded)
            if len(encoded)!=len(recal):
                raise AssertionError("different actual byte sizes after row calibration")
            cached[key]=(encoded,recal,decode(recal),cal_info["train_output_SSE_ratio"])
        return cached[key]

    def current_blobs(state):
        return {n:serialize_candidate(n,state[n]) for n in names}

    def physical_assert(raw,cal):
        if len(package(raw,1))!=GLOBAL_LIMIT_BYTES or len(package(cal,1))!=GLOBAL_LIMIT_BYTES:
            raise AssertionError("joint move violated exact physical source 343840B ceiling")
        for n in names:
            _,_,_,_,a=parse(raw[n])
            _,_,_,_,b=parse(cal[n])
            if [x[2] for x in a]!=[x[2] for x in b]:
                raise AssertionError("calibration changed frozen precision allocation")

    def rank_candidates(bits):
        # Rank proposals by TRAIN-layer real output SSE only; they are then
        # independently SELECTED via FULL MODEL TRAIN NLL in all-12 state.
        candidates=[]
        current={n:int((bits[n]==4).sum()) for n in names}
        up={};down={}
        for n in names:
            up[n]=sorted((int(i) for i in np.where(bits[n]==3)[0]),
                         key=lambda i:(-float(gain[n][i]),i))
            down[n]=sorted((int(i) for i in np.where(bits[n]==4)[0]),
                           key=lambda i:(float(gain[n][i]),i))
        seen=set()
        def append(moves,kind):
            t=tuple(sorted((n,int(row),int(target)) for n,row,target in moves))
            if not t or t in seen:
                return
            seen.add(t)
            next_counts=current.copy()
            net_bytes=0
            proxy=0.
            for n,row,target in t:
                prev=int(bits[n][row])
                if prev==target:
                    return
                net_bytes+=(target-prev)*limits[n]["row_bytes"]
                next_counts[n]+=1 if target==4 else -1
                proxy+=float(gain[n][row])*(1 if target==4 else -1)
            if net_bytes!=0 or any(not limits[n]["floor"]<=next_counts[n]<=limits[n]["ceiling"] for n in names):
                return
            candidates.append((proxy,t,kind))

        # Within-module equal-byte bit transfer tests actual model-level
        # TRAIN NLL impact of row precision, not only isolated SSE.
        for n in names:
            if up[n] and down[n]:
                append(((n,down[n][0],3),(n,up[n][0],4)),"same_module")

        # Move one 4-bit row across different matrices of the same width.
        for width in (128,512):
            donors=sorted(
                ((float(gain[n][down[n][0]]),module_identity(n),n,down[n][0])
                 for n in names if source_matrices[n].shape[1]==width
                 and current[n]>limits[n]["floor"] and down[n]),
                key=lambda x:(x[0],x[1],x[3]))[:4]
            receivers=sorted(
                ((-float(gain[n][up[n][0]]),module_identity(n),n,up[n][0])
                 for n in names if source_matrices[n].shape[1]==width
                 and current[n]<limits[n]["ceiling"] and up[n]),
                key=lambda x:(x[0],x[1],x[3]))[:4]
            for _,_,src,src_row in donors:
                for _,_,dst,dst_row in receivers:
                    if src!=dst:
                        append(((src,src_row,3),(dst,dst_row,4)),"same_width_cross_module")

        # Cross-width swap: four cheap 16-byte up-projection rows for
        # one 64-byte down-projection row, and the reverse. Both protect
        # every matrix's v38 predeclared module precision floor/ceiling.
        def choose_moves(width,prev,target,k):
            selected=[];counts=current.copy()
            entries=sorted(
                ( ((-1 if target==4 else 1)*float(gain[n][row]),
                   module_identity(n),int(row),n)
                  for n in names if source_matrices[n].shape[1]==width
                  for row in (up[n] if prev==3 else down[n]) ),
                key=lambda e:(e[0],e[1],e[2]))
            for _,_,row,n in entries:
                future=counts[n]+(1 if target==4 else -1)
                if limits[n]["floor"]<=future<=limits[n]["ceiling"]:
                    selected.append((n,row,target))
                    counts[n]=future
                if len(selected)==k:
                    return selected
            return []
        remove4short=choose_moves(128,4,3,4)
        add1long=choose_moves(512,3,4,1)
        remove1long=choose_moves(512,4,3,1)
        add4short=choose_moves(128,3,4,4)
        if remove4short and add1long:
            append(remove4short+add1long,"four_short_for_one_long")
        if remove1long and add4short:
            append(remove1long+add4short,"one_long_for_four_short")

        candidates.sort(key=lambda e:(-e[0],str(e[1]),e[2]))
        # Test top proxy proposals, but accept ONLY by actual joint TRAIN NLL.
        return candidates[:MAX_PROBES_PER_ROUND]

    start_physical=current_blobs(states)
    raw={n:start_physical[n][0] for n in names}
    calibrated={n:start_physical[n][1] for n in names}
    physical_assert(raw,calibrated)
    source_tensors={n:model.get_submodule(n).weight.detach().clone() for n in names}
    selected_train_nll=None
    history=[]
    rounds_completed=0
    best_state={n:v.copy() for n,v in states.items()}
    baseline_nll=None
    try:
        with torch.inference_mode():
            for n in names:
                model.get_submodule(n).weight.copy_(torch.from_numpy(start_physical[n][2]))
            baseline_nll=float(ppl_eval(model,train_tokens,context)["nll"])
            if not np.isfinite(baseline_nll):
                raise AssertionError("all-12 initial model TRAIN NLL is not finite")
            selected_train_nll=baseline_nll
            for step in range(MAX_ROUNDS):
                choices=rank_candidates(best_state)
                if not choices:
                    history.append({"round":step+1,"tested":0,"decision":"no_legal_moves"})
                    break
                current_round_best=None
                measurements=[]
                for proxy,moves,kind in choices:
                    touched=sorted({move[0] for move in moves},key=module_identity)
                    trial={n:best_state[n].copy() for n in touched}
                    for n,row,target in moves:
                        trial[n][row]=target
                    evidence={n:serialize_candidate(n,trial[n]) for n in touched}
                    # Physical bytes unchanged by design (moves sum 0);
                    # explicitly inspect actual packaged true bytes.
                    candidate_raw={n:(evidence[n][0] if n in evidence else raw[n])
                                   for n in names}
                    candidate_cal={n:(evidence[n][1] if n in evidence else calibrated[n])
                                   for n in names}
                    physical_assert(candidate_raw,candidate_cal)
                    try:
                        for n in touched:
                            model.get_submodule(n).weight.copy_(torch.from_numpy(evidence[n][2]))
                        trial_nll=float(ppl_eval(model,train_tokens,context)["nll"])
                        if not np.isfinite(trial_nll):
                            raise AssertionError("nonfinite joint model TRAIN NLL from proposed row swap")
                    finally:
                        for n in touched:
                            model.get_submodule(n).weight.copy_(
                                torch.from_numpy(serialize_candidate(n,best_state[n])[2]))
                    record={
                        "kind":kind,
                        "TRAIN_SSE_proxy_gain":float(proxy),
                        "actual_joint_model_train_NLL":trial_nll,
                        "delta_vs_selected_joint_train_NLL":trial_nll-selected_train_nll,
                        "byte_neutral_row_changes":[[n,int(row),int(target)] for n,row,target in moves],
                    }
                    measurements.append(record)
                    if trial_nll<selected_train_nll-MIN_ACCEPTED_NLL_REDUCTION and (
                        current_round_best is None or trial_nll<current_round_best[0]-1e-12):
                        current_round_best=(trial_nll,moves,evidence)
                rounds_completed+=1
                if current_round_best is None:
                    history.append({"round":step+1,"tested":len(measurements),
                                    "results":measurements,
                                    "decision":"no_joint_TRAIN_loss_improvement"})
                    break
                selected_train_nll,moves,winner=current_round_best
                for n,row,target in moves:
                    best_state[n][row]=target
                for n,e in winner.items():
                    raw[n],calibrated[n]=e[0],e[1]
                    model.get_submodule(n).weight.copy_(torch.from_numpy(e[2]))
                physical_assert(raw,calibrated)
                history.append({"round":step+1,"tested":len(measurements),
                                "results":measurements,
                                "decision":"accepted_lower_joint_TRAIN_NLL",
                                "accepted_moves":[[n,int(row),int(target)] for n,row,target in moves],
                                "new_full_model_train_NLL":selected_train_nll})
                print(json.dumps({"round":step+1,"candidates_measured":len(measurements),
                                  "TRAIN_NLL_before_round":baseline_nll if step==0 else history[-2].get("new_full_model_train_NLL"),
                                  "accepted":history[-1]["accepted_moves"],
                                  "best_joint_train_NLL":selected_train_nll}),flush=True)
    finally:
        with torch.no_grad():
            for n in names:
                model.get_submodule(n).weight.copy_(source_tensors[n])

    physical_assert(raw,calibrated)
    if selected_train_nll>baseline_nll+1e-9:
        raise AssertionError("selection increased joint model TRAIN NLL")
    return raw,calibrated,{
        "algorithm":"two-round frozen shortlist of 3-to-4/4-to-3 equal-byte row swaps, selected using measured FULL MODEL TRAIN NLL with ALL 12 compressed/calibrated matrices",
        "number_of_module_matrices":12,
        "original_baseline_joint_train_NLL":baseline_nll,
        "selected_joint_train_NLL":selected_train_nll,
        "joint_train_NLL_improvement":baseline_nll-selected_train_nll,
        "precision_row_bounds":{
            n:{"local_4bit":limits[n]["local"],
               "floor_4bit":limits[n]["floor"],
               "ceiling_4bit":limits[n]["ceiling"],
               "selected_4bit":int((best_state[n]==4).sum())} for n in names},
        "preregistered_max_rounds":MAX_ROUNDS,
        "preregistered_max_probes_per_round":MAX_PROBES_PER_ROUND,
        "rounds_completed":rounds_completed,
        "actual_joint_train_NLL_proposal_history":history,
        "actual_physical_binary_bytes":len(package(calibrated,1)),
        "physical_SHA256":hashlib.sha256(package(calibrated,1)).hexdigest(),
        "scoring_only_WikiText_train":True,
        "no_development_or_test_quality_used_in_selection":True,
    }
