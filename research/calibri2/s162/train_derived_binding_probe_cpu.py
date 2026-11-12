"""CALIBRI S162: frozen S161 real TRAIN-derived binding probe, no training.

S160 first real 43M/Green Tunnel shared-prefill neural two-slot head produced
456 old TRAIN 362->430 and 128 S122 TRAIN 17->127, BUT forgot 7 previously
correct original TRAIN full/EOS answers. J7 rejected S160 and restored state.
S161 reevaluates EXACT SAME frozen S160 physical FP16 head with a predeclared,
question-agnostic, answer-ID-free semantic guard:

If two distinct contiguous source spans have the same first attribute token
but a conflicting second attribute token, defer to ORIGINAL S142 KV decoder.
All spans must come from the shared original prompt; no expected answer,
gold IDs, question identity, original correctness, QA/TEST or empirical
failure-list lookup is allowed in the deployed selector.

This is an actual CPU full/EOS 456+128, FP16 physical J7 candidate using
original frozen S142 and original physical S160 GPU weights. No more MPS
training, no release or promotion, no LIVE, Render/Supabase or APK.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import cohort,load_cpu_original
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import load_compact_q2
from calibri2.s132.progressive_core import ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY
from calibri2.s144.train_one_continued_j7 import frozen_routes
from calibri2.s158 import targeted_retention_gpu as s158
from calibri2.s159.parallel_slot_binding import ParallelSlotBindingHead
from calibri2.s160 import real_parallel_slots_j7 as s160

STAGE="S162_TRAIN_DERIVED_BINDING_PROBE_S142_J7"
PHYSICAL_SHA="INVALID_UNTIL_READ_FROM_PINNED_S160_AUDIT"
COUNTERS=Counter()
ORIGINAL_SELECTOR=s160.select_parallel


def ambiguity_guarded_parallel(head, hidden, ids, tok):
    """No gold / identity / parent outcome. Shared-source conflicting attribute.

    A single shared-prefix difference is the ambiguity criterion: two
    different second attributes from the same first attribute. The current
    selected source span must itself belong to that conflict set.
    Conservative route: original S142 decoder, never a confident copied noun
    from a potentially wrong owner.
    """
    choice=ORIGINAL_SELECTOR(head,hidden,ids,tok)
    if choice is None:
        COUNTERS["S160_confidence_fallback"]+=1
        return None
    canonical=s160.normalized_candidate_spans(tok,ids[0].tolist())
    other_exists=any(
        int(row[0])==int(choice[0]) and int(row[1])>=0 and
        int(row[1])!=int(choice[1])
        for row in canonical
    )
    if other_exists:
        COUNTERS["S161_same_attribute_conflict_parent_fallback"]+=1
        return None
    COUNTERS["S161_unambiguous_parallel_selected"]+=1
    return choice



def make_train_derived_probe(tok, old, new):
    """A NEW deterministic 32-pair TRAIN-derived diagnostic, not embargoed QA.

    Source material: original S82 TRAIN names/frames/24 words and S30 TRAIN
    colors ONLY. No new vocab, QA/TEST, validation outputs, or learned updates.
    The S160 neural head never trained on these exact 64 question hashes.
    Half the pairs have same-color conflicting nouns: a useful hard case for
    the S161 semantic guard. Remaining pairs have different colors.
    """
    from calibri2.s82 import real_mps_generic_token_context_copy as s82
    from v186.pairs import tokenize_one
    _, words = s82.newly_authored_train(tok)
    colors = tuple(sorted(s30.TRAIN_ELIGIBLE))
    if len(words) != 24 or len(colors) != 4:
        raise AssertionError("S162 original TRAIN vocabulary ABI drift")
    historical = {sha(x["question"].encode()) for x in (*old.values(), *new.values())}
    if len(historical) != 584:
        raise AssertionError("S162 original 456+128 TRAIN source is not unique")
    generated, specs, seen = {}, [], set()
    for i in range(32):
        frame_index = i % 4
        a, b = s82.TRAIN_NAMES[frame_index]
        frame = s82.TRAIN_FRAMES[frame_index]
        first_color = colors[(i * 3 + 1) % 4]
        second_color = (first_color if i % 2 == 0
                        else colors[(colors.index(first_color) + 1 + i // 8 % 2) % 4])
        noun_a = words[(i * 11 + 7) % 24]
        answer_a = f"{first_color} {noun_a}"
        for offset in range(24):
            noun_b = words[(i * 13 + 17 + offset) % 24]
            if noun_b == noun_a:
                continue
            answer_b = f"{second_color} {noun_b}"
            sides = [
                ("left", frame.format(a=a, b=b, left=answer_a, right=answer_b), answer_a),
                ("right", frame.format(a=a, b=b, left=answer_b, right=answer_a), answer_b),
            ]
            samples = [tokenize_one(tok, question, answer)
                       for _, question, answer in sides]
            digests = [sha(question.encode()) for _, question, _ in sides]
            if (len(set(digests)) != 2
                or any(x in historical or x in seen for x in digests)
                or any(len(tok.encode(sample["answer"], add_special_tokens=False)) != 2
                       for sample in samples)):
                continue
            for (side, question, answer), sample, digest in zip(sides, samples, digests):
                expected = tok.encode(answer, add_special_tokens=False)
                if (expected[0] != sample["first_expected_id"]
                    or tok.eos_token_id in expected):
                    raise AssertionError("S162 original two-token TRAIN ABI changed")
                key = f"S162_TRAIN_DERIVED/{162000+i}/{side}"
                generated[key] = {
                    "id": key, "pair_id": 162000+i, "side": side,
                    "question": question, "answer": answer,
                    "expected": expected + [tok.eos_token_id],
                    "source": "TRAIN-derived diagnostic; never used for MPS head fitting",
                }
                seen.add(digest)
            specs.append({"pair": 162000+i, "same_color": i % 2 == 0})
            break
        else:
            raise AssertionError("S162 deterministic TRAIN-derived pair construction failed")
    if len(generated) != 64 or len(specs) != 32 or len(seen) != 64:
        raise AssertionError("S162 expected 64 distinct previously-unseen TRAIN derivatives")
    return generated, {
        "source": "only original S30/S82 TRAIN lexicon and templates",
        "not_independent_QA_TEST": True,
        "not_a_heldout_generalization_claim": True,
        "never_used_for_S160_training": True,
        "unique_disjoint_from_original_584_TRAIN_question_hashes": True,
        "pair_count": 32,
        "question_count": 64,
        "same_color_different_noun_pairs": 16,
        "different_color_pairs": 16,
        "sha256_of_sorted_question_hashes": sha("\n".join(sorted(seen)).encode()),
    }


def main():
    # Identical canonical verified S158 source CLI, plus a *pinned* unaccepted
    # S160 FP16 physical artifact and pinned S160 report before output path.
    rawargs=list(sys.argv)
    if len(rawargs)<4:
        raise AssertionError("S161 missing pinned S160 physical inputs")
    s160_fp16=Path(rawargs[-3]);s160_audit=Path(rawargs[-2])
    sys.argv=rawargs[:-3]+[rawargs[-1]]
    args=s158.args_from_cli()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    parent_guard,side,q2,inherited=s158.check_sources(args)
    trial=json.loads(s160_audit.read_text())
    if (trial.get("stage")!="CALIBRI_S160_FIRST_REAL_43M_PARALLEL_SLOT_MPS_EXPERIMENT"
        or trial.get("safe_research_candidate") is not False
        or trial.get("J7_rollback_verified") is not True
        or trial.get("retention_456_128",{}).get("old_full_before")!=362
        or trial.get("retention_456_128",{}).get("old_full_after")!=430
        or trial.get("retention_456_128",{}).get("new_full_before")!=17
        or trial.get("retention_456_128",{}).get("new_full_after")!=127
        or len(trial.get("retention_456_128",{}).get("lost_old_full_ids",[]))!=7
        or trial.get("physical_FP16_sha256")!=sha(s160_fp16.read_bytes())
        or trial.get("S142_physical_parent_sha256")!=sha(args.S142_parent_sidecar.read_bytes())
        or trial.get("embargoed_QA_never_run") is not True):
        raise AssertionError("S161 requires exact rejected physical S160 real MPS artifact")
    head=ParallelSlotBindingHead(320,slots=2,rank=s160.SLOT_RANK).eval()
    reloaded=load_file(str(s160_fp16))
    if any(x.dtype!=torch.float16 for x in reloaded.values()):
        raise AssertionError("S161 inherited S160 must be real persisted FP16")
    head.load_state_dict({k:v.float() for k,v in reloaded.items()},strict=True)
    if any(p.requires_grad for p in head.parameters()):
        head.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    novel,digest,_=materialize_train_only(tok)
    if (digest!=json.loads(args.S122_curriculum_report.read_text())["new_TRAIN_question_hashset_sha256"]
        or len(novel)!=128):
        raise AssertionError("S161 original S122 TRAIN source changed")
    new={row["id"]:row for row in novel}
    new_ids=sorted(new)
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(args)
    new_routes=frozen_routes(tok,router,new)
    reports={key:freeze_report(getattr(args,key+"_report")) for key in ("S87","S91","S93")}
    old,old_ids,old_routes=cohort(tok,reports,router)
    routes={**old_routes,**new_routes}
    if len(old_ids)!=456 or len(new_ids)!=128:
        raise AssertionError("S161 must audit all original TRAIN examples")
    raw=s87.open_envelope(args.S93_CB61.read_bytes())
    cpu=load_cpu_original(raw,side)
    bridge=full=expansion=None
    s160.select_parallel=ambiguity_guarded_parallel
    try:
        bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(cpu[0])
        bridge.coefficients.requires_grad_(False)
        full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        load_compact_q2(full,q2)
        full.install(cpu[0])
        expansion=(ProgressiveRouteCore(rank=896).eval()
                   .bind_original_route_bridge(bridge).install(cpu[0]))
        expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],
                                    WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
        parent_old,parent_new=s158.run_outputs(
            cpu,old,new,old_ids,new_ids,old_routes,new_routes,
            tok,bank,copy_net,bridge,full,expansion)
        if (sum(parent_old[k]==old[k]["expected"] for k in old)!=362
            or sum(parent_new[k]==new[k]["expected"] for k in new)!=17):
            raise AssertionError("S161 immutable parent S142 checkpoint mismatch")
        pred_old,pred_new,active,examples=s160.audit_cpu_outputs(
            cpu,old,new,old_ids,new_ids,routes,tok,bank,copy_net,
            bridge,full,expansion,parent_old,parent_new,head)
        # No retraining, no rethresholding: S161 selector stays bit-for-bit frozen.
        # Probe labels are used only AFTER both parent/candidate generations.
        probe, probe_meta = make_train_derived_probe(tok,old,new)
        probe_ids = sorted(probe)
        probe_routes = frozen_routes(tok,router,probe)
        _, parent_probe = s158.run_outputs(
            cpu,{},probe,[],probe_ids,{},probe_routes,
            tok,bank,copy_net,bridge,full,expansion)
        _, pred_probe, probe_activity, probe_examples = s160.audit_cpu_outputs(
            cpu,{},probe,[],probe_ids,probe_routes,tok,bank,copy_net,
            bridge,full,expansion,{},parent_probe,head)
        probe_retention = s158.retention(
            {},{},{},parent_probe,pred_probe,probe)
        probe_loss_free = not any(probe_retention[k] for k in (
            "lost_new_full_ids","lost_new_first_ids"))
        probe_gain = (
            probe_retention["new_full_after"] > probe_retention["new_full_before"]
            or probe_retention["new_pairs_both_after"] >
            probe_retention["new_pairs_both_before"])
        retention=s158.retention(parent_old,pred_old,old,parent_new,pred_new,new)
        strict=not any(retention[k] for k in (
            "lost_old_full_ids","lost_old_first_ids",
            "lost_new_full_ids","lost_new_first_ids"))
        gain=(retention["old_full_after"]>362 or retention["new_full_after"]>17
              or retention["new_pairs_both_after"]>0)
        speed=s160.paired_cpu_ratio(
            cpu,old,old_ids,routes,tok,bank,copy_net,
            bridge,full,expansion,parent_old,head)
        # Physical S160 weight file reused byte-for-byte, no additional
        # learned parameters. S161 runtime metadata is small but counted.
        runtime=args.out/"S161_SEMANTIC_AMBIGUITY_RUNTIME.json"
        runtime.write_text(json.dumps({
            "stage":STAGE,"parent_sha":sha(args.S142_parent_sidecar.read_bytes()),
            "S160_FP16_sha256":sha(s160_fp16.read_bytes()),
            "route":"S93","same_first_different_second_conflict_parent_fallback":True,
            "immutable_FP16":True,"default_enabled":False,
            "original_decoding_on_ambiguity":True,"model_accepted":False,
            "no_QA_no_LIVE":True},sort_keys=True))
        probe_runtime=args.out/"S162_TRAIN_DERIVED_PROBE_POLICY.json"
        probe_runtime.write_text(json.dumps({
            **probe_meta, "stage": STAGE, "S160_neural_head_SHA256": sha(s160_fp16.read_bytes()),
            "S161_semantic_guard_unchanged": True, "new_gradient_steps": 0,
            "model_accepted": False, "LIVE_untouched": True
        },ensure_ascii=False,sort_keys=True))
        baseline_params=43_261_760
        physical_bpp=(parent_guard["physical_bpp"]*baseline_params
                      +8*(s160_fp16.stat().st_size+runtime.stat().st_size+probe_runtime.stat().st_size)
                      )/(baseline_params+sum(p.numel() for p in head.parameters()))
        safe=(strict and gain and probe_loss_free and probe_gain and physical_bpp<=0.7 and speed["ratio"]>=0.9)
        # Any failure never changes model: S160 physical sidecar remains a
        # rejected immutable research artifact; no optimizer created/changed.
        if not safe:
            if sha(s160_fp16.read_bytes())!=trial["physical_FP16_sha256"]:
                raise AssertionError("S161 rejected physical S160 candidate mutated")
        report={
            "stage":STAGE,"S142_unchanged_SHA256":sha(args.S142_parent_sidecar.read_bytes()),
            "S160_rejected_FP16_reused_SHA256":sha(s160_fp16.read_bytes()),
            "S160_original_seven_old_full_losses":7,
            "selection_policy":"conflicting source two-token spans sharing attribute-1 -> parent",
            "new_training_steps":0,"QA_TEST_never_run":True,"LIVE_untouched":True,
            "J7_retention":retention,"cpu_ratio":speed,
            "train_derived_probe_policy":probe_meta,
            "train_derived_probe_retention":probe_retention,
            "train_derived_probe_activation":probe_activity,
            "train_derived_probe_examples":probe_examples,
            "probe_loss_free":probe_loss_free,
            "probe_has_genuine_untrained_question_gain":probe_gain,
            "all_in_bpp":physical_bpp,"candidate_activity":active,
            "generic_gate_counts":dict(COUNTERS),"actual_TRAIN_examples":examples,
            "safe_research_candidate":safe,"model_accepted":False,
            "original_S142_unmodified":True,
        }
        (args.out/"S162_REAL_TRAIN_DERIVED_64_QUESTION_PROBE.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
        print(json.dumps({
            "event":"S162_REAL_TRAIN_DERIVED_BINDING_64_QUESTION_AUDIT",
            "old_before":362,"old_after":retention["old_full_after"],
            "new_before":17,"new_after":retention["new_full_after"],
            "lost_old":len(retention["lost_old_full_ids"]),
            "lost_new":len(retention["lost_new_full_ids"]),
            "parallel_kept":active.get("parallel_accepted",0),
            "ambiguity_fallback":COUNTERS["S161_same_attribute_conflict_parent_fallback"],
            "probe_before":probe_retention["new_full_before"],
            "probe_after":probe_retention["new_full_after"],
            "probe_lost_full":len(probe_retention["lost_new_full_ids"]),
            "probe_lost_first":len(probe_retention["lost_new_first_ids"]),
            "probe_pairs_both_before":probe_retention["new_pairs_both_before"],
            "probe_pairs_both_after":probe_retention["new_pairs_both_after"],
            "probe_parallel_kept":probe_activity.get("parallel_accepted",0),
            "bpp":physical_bpp,"cpu_ratio":speed["ratio"],
            "safe":safe}),flush=True)
    finally:
        s160.select_parallel=ORIGINAL_SELECTOR
        if expansion is not None:expansion.uninstall()
        if full is not None:full.uninstall()
        if bridge is not None:bridge.uninstall()
        cpu[6].uninstall()


if __name__=="__main__":
    main()
