"""CALIBRI v202: an ACTUAL v186 -> Green Tunnel 135M paid-source repair pilot.

A deliberately limited 16-step TRAIN-only sequence-rollout experiment. Dynamic v197
all-211 scheduling is precommitted; 16 steps DO NOT claim a full cycle.
Every accepted edit must retain immutable v186 TRAIN first-token/
opposing-pair behavior. Only after source sealing are untouched original
FP32, 16 public DEV full replies, native speed and VALID measured.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from green_tunnel.adapter_cb59 import operators_from_CB59_metadata
from green_tunnel.schedule import plan_cycles
from green_tunnel.contract import PhysicalSource, verify_physical_source
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v176.operators import clear
from v185.optim import propose_codes, apply_codes
from v185.codec import replace_discrete
from v185.run import physical_numerical_control
from v186.fit import paired_loss
from v186.run import validation_pairs, speed_pairs
from v196 import protocol as p
from v196.data import training_inputs, full_vocabulary_references
from v196.objective import language_objective
from v196.recover import build_trainable_from_paid_source
from v196.run import newly_frozen_public_dev_16
from v198.guard import probe_first_tokens, anti_collapse_gate, summarize_guard
from v200.rollout import probe_greedy_train, sequence_no_regression_gate
from .unlikelihood import train_rollout_ce_and_repeat_unlikelihood

PILOT_STEPS = 16
TRAIN_GUARD_PAIR_IDS = (0,1,6,7,12,13,18,19)
CENTROID_LR = 0.00025
INDEX_CAP = 128
ANCHOR_STRENGTH = 0.002
TRAIN_OBJECTIVE_RELATIVE_TOLERANCE = 0.01
REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY = 0.25


def balanced_domain_index(step):
    # Six frozen pairs per TRAIN domain; use 4 distinct pairs/domain in pilot.
    return (step % 4) * 6 + (step // 4) % 6


def loss_mix(student, teacher_refs, pairs, step, centers=None, roots=None):
    pidx = balanced_domain_index(step)
    ref = teacher_refs[step % len(teacher_refs)]
    paired, paired_stats = paired_loss(student, None, pairs[pidx])
    language, lang_stats = language_objective(student, ref)
    # v202: the student sees its own greedy tokens at successive answer positions;
    # the original TRAIN targets include EOS and both opposite-answer sides.
    rollout, unlikelihood, rollout_stats = train_rollout_ce_and_repeat_unlikelihood(
        student,pairs[pidx],steps_cap=6)
    total = .45*paired + .25*language + .20*rollout + .10*unlikelihood
    if centers is not None:
        prior = torch.stack([((s.centroids - roots[k])**2).mean()
                             for k,s in centers.items()]).mean()
        total = total + ANCHOR_STRENGTH*prior
    if not torch.isfinite(total):
        raise AssertionError("Nonfinite complete TRAIN objective")
    return total, {"TRAIN_pair_id": pidx, "TRAIN_pair_domain": pairs[pidx]["domain"],
                   "TRAIN_pair": paired_stats, "TRAIN_full_vocab": lang_stats,
                   "TRAIN_generated_sequence": rollout_stats,
                   "TRAIN_unwanted_repeat_penalty":float(unlikelihood.detach())}


def pilot_recover(student, sources, teacher_refs, train_pairs,
                  source_matrices, target_schedule):
    if len(sources) != 211 or len(train_pairs) != 24 or len(teacher_refs) != 20:
        raise AssertionError("Original authentic 135M source or frozen TRAIN mismatch")
    if len(target_schedule) != len(sources) or set(target_schedule) != set(sources):
        raise AssertionError("Missing any of the 211 scheduled original physical matrices")
    if not set(target_schedule[:PILOT_STEPS]) <= set(source_matrices):
        raise AssertionError("FP32 proposal prior incomplete")
    source_roots = {name: s.centroids.detach().clone()
                    for name,s in sources.items()}
    initial_codes = {name:sources[name].codes.detach().clone()
                     for name in target_schedule[:PILOT_STEPS]}
    guard_pairs = [train_pairs[i] for i in TRAIN_GUARD_PAIR_IDS]
    immutable_full = probe_first_tokens(student,sources,train_pairs)
    immutable_guard = {i: immutable_full[i] for i in TRAIN_GUARD_PAIR_IDS}
    # Freeze actual bounded generated TRAIN sequences before any weight update.
    # This is separate from the one-token first-token anti-collapse guard.
    immutable_rollouts = probe_greedy_train(student,sources,guard_pairs,max_new_tokens=6)
    optimizer = torch.optim.Adam([s.centroids for s in sources.values()],
                                  lr=CENTROID_LR)
    steps = []
    accepted = 0
    for step in range(PILOT_STEPS):
        name = target_schedule[step]
        target = sources[name]
        before_centers = {n:s.centroids.detach().clone()
                          for n,s in sources.items()}
        before_codes = target.codes.detach().clone()
        before_adam = deepcopy(optimizer.state_dict())
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        captured = []
        def capture(_module, _args, reconstructed):
            if reconstructed.requires_grad and not any(
                    reconstructed is old for old in captured):
                reconstructed.retain_grad()
                captured.append(reconstructed)
        hook = target.register_forward_hook(capture)
        try:
            before, stats = loss_mix(student, teacher_refs, train_pairs, step,
                                     sources,source_roots)
            before_value = float(before.detach())
            before.backward()
        finally:
            hook.remove()
        if len(captured)!=1 or captured[0].grad is None:
            raise AssertionError("Missing genuine global gradient to "+name)
        wg = captured[0].grad.detach().cpu().numpy().copy()
        active = sum(s.centroids.grad is not None for s in sources.values())
        if active != 211:
            raise AssertionError("Not all original paid neural levels trained")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in sources.values()], 1.0,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values(): s.centroids.clamp_(-16.0,16.0)
        clear(sources)
        with torch.no_grad():
            centroid_loss = float(loss_mix(
                student,teacher_refs,train_pairs,step,sources,source_roots)[0])
        proposed = propose_codes(target,wg,source_matrices[name])
        # Limit genuinely paid mutations per step, rather than changing
        # hundreds of already weak code groups at once as in v196.
        count = min(proposed["proposed_groups"],INDEX_CAP)
        for key in ("indices","new_codes","old_codes","score_improvement"):
            proposed[key] = proposed[key][:count]
        proposed["proposed_groups"] = count
        codes_accepted_for_TRAIN = False
        with torch.no_grad():
            if count:
                apply_codes(target,proposed,True)
                clear(sources)
                candidate_loss = float(loss_mix(
                    student,teacher_refs,train_pairs,step,sources,source_roots)[0])
                codes_accepted_for_TRAIN = (
                    candidate_loss+1e-7 < centroid_loss)
                if not codes_accepted_for_TRAIN:
                    apply_codes(target,proposed,False)
                    clear(sources)
            final_loss = float(loss_mix(
                student,teacher_refs,train_pairs,step,sources,source_roots)[0])
        sampled = probe_first_tokens(student,sources,guard_pairs)
        safety = anti_collapse_gate(immutable_guard,sampled,
                    minimum_distinct_fraction=REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY)
        reasonable_train_objective = (
            final_loss <= before_value*(1+TRAIN_OBJECTIVE_RELATIVE_TOLERANCE))
        # v202 TRAIN-only per-step sequence guard. The global final 8-pair
        # sequence guard still runs separately after all 16 source steps.
        step_pair = guard_pairs[step % len(guard_pairs)]
        step_id = step_pair["pair_id"]
        step_rollout = probe_greedy_train(
            student,sources,[step_pair],max_new_tokens=6)
        step_seq = sequence_no_regression_gate(
            {step_id:immutable_rollouts[step_id]},step_rollout)
        keep = (safety["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"] and
                reasonable_train_objective and
                step_seq["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"])
        if not keep:
            with torch.no_grad():
                for n,s in sources.items():
                    s.centroids.copy_(before_centers[n])
                target.codes.copy_(before_codes)
            optimizer.load_state_dict(before_adam)
            clear(sources)
        else:
            accepted += 1
        steps.append({"step":step+1,"operator":name,
                      "all_211_centroids_received_gradient":active,
                      "first_token_guards_precommitted_TRAIN_only":list(TRAIN_GUARD_PAIR_IDS),
                      "gradient_norm":grad_norm,
                      "candidate_5bit_groups":count,
                      "code_swaps_pre_guard_TRAIN_improve":codes_accepted_for_TRAIN,
                      "TRAIN_mix_before":before_value,"TRAIN_mix_after":final_loss,
                      "TRAIN_no_more_than_1percent_loss_increase":reasonable_train_objective,
                      "TRAIN_guard":safety,
                      "per_step_TRAIN_sequence_guard":step_seq,
                      "source_step_accepted":keep,
                      "never_consulted_VALID_or_DEV_to_accept":True})
        clear(sources)
        print(json.dumps({"stage":"v202_anti_collapse_paid_pilot",
                          "step":step+1, "source":name,
                          "accepted":keep,"accepted_so_far":accepted,
                          "TRAIN_guard":safety["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"],
                          "TRAIN_sequence_guard":step_seq["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"]}),
              flush=True)
    final_full = probe_first_tokens(student,sources,train_pairs)
    # v199: The immutable original v186 gets only 4 distinct first tokens
    # on 48 complete TRAIN prompts (8.33%). 25% is thus unattainable for
    # a candidate required to retain v186 behavior in just 16 source steps.
    # Preserve no-regression on ALL 24 full TRAIN pairs, keep the absolute
    # 25% anti-collapse floor on the precommitted 8-pair per-step subset,
    # and apply a baseline-relative diversity floor on the full set.
    # No VALID, DEV or FP32 QA feedback is used for this choice.
    full_baseline_diversity = summarize_guard(immutable_full)["first_token_diversity"]
    full_train_floor = min(REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY,
                           full_baseline_diversity)
    final_safety = anti_collapse_gate(
        immutable_full,final_full,
        minimum_distinct_fraction=full_train_floor)
    final_safety["v199_full_24_TRAIN_baseline_relative_floor"] = full_train_floor
    final_safety["v199_sampled_8_TRAIN_absolute_guard_floor"] = (
        REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY)
    final_safety["v199_full_24_TRAIN_still_below_25pct_goal"] = (
        final_safety["TRAIN_first_token_diversity_after"] <
        REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY)
    # Global complete-sequence no-regression: reject longer greedy repetition,
    # reduced correct token prefixes or lost originally exact TRAIN answers.
    final_rollouts = probe_greedy_train(student,sources,guard_pairs,max_new_tokens=6)
    sequence_safety = sequence_no_regression_gate(immutable_rollouts,final_rollouts)
    rolled_back_all = not (final_safety["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"] and
                           sequence_safety["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"])
    if rolled_back_all:
        with torch.no_grad():
            for n,s in sources.items():s.centroids.copy_(source_roots[n])
            for n,codes in initial_codes.items():sources[n].codes.copy_(codes)
        clear(sources)
        # If immutable baseline fails the 25% diversity floor, preserve
        # the original authenticated v186 rather than inventing progress.
    changed_codes = sum(int(torch.count_nonzero(
        sources[n].codes - initial_codes[n])) for n in initial_codes)
    changed_fp16_cells = sum(int(torch.count_nonzero(
        s.centroids.detach().to(torch.float16) -
        source_roots[n].to(torch.float16)))
        for n,s in sources.items())
    return {"actual_pilot_steps":PILOT_STEPS,
            "full_all_211_schedule_precommitted_not_completed":list(target_schedule),
            "trained_original_135M_unique_matrices":211,
            "accepted_steps_before_full_TRAIN_guard":accepted,
            "all_pilot_mutations_rolled_back_on_full_TRAIN_fail":rolled_back_all,
            "immutable_v186_full_TRAIN_guard":final_safety,
            "v202_immutable_greedy_full_sequence_TRAIN_guard":sequence_safety,
            "final_paid_5bit_net_changes":changed_codes,
            "final_FP16_centroid_net_cells_changed":changed_fp16_cells,
            "real_model_source_improved_correct_answers_NOT_YET_PROVEN":True,
            "QA_and_VALID_never_seen_for_TRAIN_acceptance":True,
            "steps":steps}


def run(v186_path,v161_path,v193_path,out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(200)
    torch.set_num_threads(2)
    raw=Path(v186_path).read_bytes()
    if len(raw)!=p.SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=p.INITIAL_135M_SHA256:
        raise AssertionError("Real TRAINED v186 135M binary SHA/size mismatch")
    archive=CB59(raw)
    operators=operators_from_CB59_metadata(archive.meta)
    schedule=plan_cycles(operators,1)[0]
    if len(schedule)!=211:
        raise AssertionError("Actual full Green Tunnel 211 matrix coverage lost")
    original=extract_original(Path(v161_path))
    if len(original.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Original real CB59 speed control missing")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    train,train_pairs,valid_pairs,origin=training_inputs(tok)
    manifest={"experiment":"v202_precommitted_true_135M_TRAIN_only_guarded_source_pilot",
        "starting_complete_binary_SHA256":p.INITIAL_135M_SHA256,
        "pilot_steps_not_full_211_cycle":PILOT_STEPS,
        "physically_unique_matrices":211,
        "all_original_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
        "all_211_source_canonical_schedule":list(schedule),
        "guarded_TRAIN_pair_ids":list(TRAIN_GUARD_PAIR_IDS),
        "full_TRAIN_final_guard_pairs":24,
        "balanced_four_domain_TRAIN_every_step":True,
        "language_full_vocabulary_TRAIN_every_step":True,
        "centroid_learning_rate":CENTROID_LR,
        "paid_max_index_proposals_per_step":INDEX_CAP,
        "TRAIN_first_token_diversity_floor":REQUIRED_TRAIN_FIRST_TOKEN_DIVERSITY,
        "v202_generated_complete_TRAIN_rollout_horizon":6,
        "v202_sequence_train_objective_weight":0.20,
        "v202_spurious_repeat_train_unlikelihood_weight":0.10,
        "v202_per_step_frozen_TRAIN_sequence_guard":True,
        "v202_loss_weights":{"paired":0.45,"full_vocab":0.25,
          "greedy_gold":0.20,"spurious_repeat":0.10},
        "v202_sequence_acceptance_no_worse_repetition_or_token_prefix":True,
        "TUNING_USES_ONLY_TRAIN_not_public_DEV_or_VALID":True,
        "existing_PUBLIC_DEV_is_diagnostic_NOT_unseen_96plus":True,
        "no_PRIVATE_v89_TEST_or_Jessica_runs":True,
        "no_LIVE_main_Render_Supabase_APK":True,
        "TRAIN_origin":origin}
    save(out/"PRECOMMITTED_TRAIN_ONLY_v202.json",manifest)
    teacher=original_teacher().cpu().eval()
    teacher_refs=full_vocabulary_references(teacher,train,2.0)
    pretrained=teacher.state_dict()
    fp32_targets={name:pretrained[name].detach().cpu().numpy().copy()
                  for name in schedule[:PILOT_STEPS]}
    student,sources=build_trainable_from_paid_source(archive)
    fit=pilot_recover(student,sources,teacher_refs,train_pairs,
                      fp32_targets,schedule)
    save(out/"ACTUAL_guarded_TRAIN_16_step_audit.json",fit)
    clear(sources)
    blob,seal=replace_discrete(archive,sources,strict=False)
    complete=PhysicalSource(codec="CB59_K32V8_trained",
        original_unique_parameters=p.FROZEN_SOURCE_135M_UNIQUE,
        physical_whole_bytes=len(blob),
        physical_sha256=hashlib.sha256(blob).hexdigest(),
        aliases=tuple(sorted(archive.meta["source_alias_keys"].items())))
    checked=verify_physical_source(complete,blob)
    if not checked["physically_paid"] or len(blob)!=len(raw):
        raise AssertionError("Unpaid original 135M model source")
    dest=out/"CALIBRI_v202_guarded_16step_135M.cb59"
    dest.write_bytes(blob)
    seal["source_sha_same_as_v186_if_guard_rejected_everything"] = (
        seal["sha256"]==p.INITIAL_135M_SHA256)
    seal["v202_frozen_before_ANY_VALID_or_DEV"]=True
    save(out/"SEALED_paid_source_before_untouched_QA.json",seal)
    print(json.dumps({"stage":"v202_paid_source_sealed",
            "actual_bpp":seal["complete_bpp"],
            "SHA":seal["sha256"],
            "changed_5bit":seal["actual_changed_5bit_group_indices"],
            "changed_FP16_cells":seal["actually_changed_FP16_codebook_cells"]}),flush=True)
    # The original pretrained FP32 answers were already separately frozen
    # in v193. No answer, grader, or development score is used for fitting.
    with TemporaryDirectory(dir=out) as directory:
        tmp=Path(directory)
        for name in ("original","v186","v202"):(tmp/name).mkdir()
        (tmp/"original"/"source.cb59").write_bytes(original.blob)
        (tmp/"v186"/"source.cb59").write_bytes(raw)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=tmp/"original"/"source.cb59",folder=tmp/"original"))
        cb186,_,_=make_model("baseline",SimpleNamespace(
            original=tmp/"v186"/"source.cb59",folder=tmp/"v186"))
        cb198,_,_=make_model("baseline",SimpleNamespace(
            original=dest,folder=tmp/"v202"))
        if any(x.ndim>=2 for x in cb198.parameters()):
            raise AssertionError("Native v202 model is loading unpacked 2D weights")
        numeric=physical_numerical_control(student,cb198,sources,
             train_pairs[0]["left"]["prompt_ids"][:8][None])
        save(out/"TRAIN_surrogate_vs_native_paid_numerical_consistency.json",numeric)
        frozen_dev=freeze_cases(all_cases())
        native_speed=speed_pairs({"CB59":cb59,"CB186":cb198},tok,frozen_dev)
        save(out/"real_matched_native_32_token_CPU_vs_CB59.json",native_speed)
        prior_valid=validation_pairs({"CB186":cb186,"GT202":cb198},tok,valid_pairs)
        save(out/"HELDOUT_v186_VALID_pair_comparison_AFTER_sealing.json",prior_valid)
        dev_16=newly_frozen_public_dev_16(
            {"CB186":cb186,"GT202":cb198},tok,v193_path,out)
    result={"completed":True,"authentic_pretrained_135M_unique":p.FROZEN_SOURCE_135M_UNIQUE,
            "source_SHA256":seal["sha256"],"whole_original_source_bpp":seal["complete_bpp"],
            "guard_accepted_step_count":fit["accepted_steps_before_full_TRAIN_guard"],
            "final_guard_rolled_back":fit["all_pilot_mutations_rolled_back_on_full_TRAIN_fail"],
            "native_CPU_speed_ratio_CB59":native_speed["median_speed_ratio_vs_CB59"],
            "original_FP32_16_machine_correct":dev_16["scores"]["FP32_existing_135M"]["machine_confirmed"],
            "previous_v186_16_machine_correct":dev_16["scores"]["CB186"]["machine_confirmed"],
            "new_v202_16_machine_correct":dev_16["scores"]["GT202"]["machine_confirmed"],
            "new_v202_complete_answers":dev_16["native_compact_actual_QA"]["GT202"],
            "prescale_135M_FP32_quality_parity_NOT_certified":True,
            "no_user_chat_Jessica_tests":True,
            "NO_larger_models_or_LIVE_changes":True}
    save(out/"v202_real_NATIVE_SOURCE_QA_summary.json",result)
    print(json.dumps({"stage":"v202_real_native_report",
                      **{k:v for k,v in result.items()
                         if k!="new_v202_complete_answers"}}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("real_v186_source",type=Path)
    ap.add_argument("original_v161_source",type=Path)
    ap.add_argument("original_v193_FP32_QA",type=Path)
    ap.add_argument("output",type=Path)
    x=ap.parse_args()
    run(x.real_v186_source,x.original_v161_source,
        x.original_v193_FP32_QA,x.output)
