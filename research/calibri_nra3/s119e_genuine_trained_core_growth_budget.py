"""S119e real-parent audited NUMERICAL growth ledger for NRA3-R/Calibri.

USER REQUIREMENTS NEVER CHANGE: <=0.700 bpp all-in, full CPU >=0.90x
target / >=0.85x absolute floor, all original 616 J7 skills unchanged,
positive independent understanding. Experimental >0.7 is permitted
ONLY in frozen shadow, never accepted/activated.

IMPORTANT: increase in architecture's "parameter count" does NOT
magically reduce real compressed size. Only genuinely trained, functional,
uniquely counted, physically encoded parameters can enter the denominator
of a future acceptance audit. Every byte of growth (and runtime codec,
metadata, GT, NRA3, adapters) enters the numerator. Filling zero/fixed
S132 seed-derived channels cannot be used to manufacture a low bpp.
Existing zero/fixed structural parameters were ALREADY present in
S194's 43,261,760 legacy denominator and cannot be added a second time.

S119e consumes only previous S119d 37,087 physically saved child and report;
this is a forecast with ACTUAL previous parent size and real child bytes,
NOT a compressed learned-growth checkpoint or a measured performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, json, math
from pathlib import Path

from research.calibri_nra3.s119a_hard_four_goals import (
    BPP_MAX, CPU_MIN, CPU_TARGET, SOURCE_UNIQUE_TRAINED_WEIGHTS, check,
)
from research.calibri_nra3.s119b_S194_parent_S115_child_physical_budget import (
    S194_REFERENCE_DENOM, historical_projection, S115_EXPECTED_SHA256,
)
# Independent SHA pins: S119e is intentionally stdlib-only; importing the
# S119d torch module would expand the runtime and potentially open labels.
S119D_REPORT = "NRA3_S119D_REAL12_TRAIN_A_NRA2_MINI8_SHARED_GRAPH_AND_BUDGET.json"
S119D_CHILD = "NRA3_S119D_SHARED_EXISTING_37087_FP16_NO_NEW_MATRIX_UNACCEPTED.safetensors"
PARENT_H_SHA = "e5b91042eae7311226b693222c557898caca27d6a3ec6f029574714a27f6d7ec"

STAGE="NRA3_S119E_AUDITED_REAL_S119D_PARENT_GENUINE_CORE_GROWTH_BPP_LEDGER_RESEARCH_ONLY"
REPORT="NRA3_S119E_REAL_S119D_PARENT_GENUINE_CORE_GROWTH_SCENARIOS_NOT_ACCEPTED.json"
LEGACY_ARCHITECTURE_CORE_PARAMS=S194_REFERENCE_DENOM
UNTRAINED_OR_FIXED_ARCHITECTURE_GAP=(
    S194_REFERENCE_DENOM-SOURCE_UNIQUE_TRAINED_WEIGHTS)
MODES=("new_trained_core","activate_old_architecture_training")


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_growth(
    *, parent_charged_byte_equivalent: float, grow_unique_trained_params: int,
    new_packed_bytes: int, growth_mode: str,
    newly_functional_trained_params: int, counts_verified_by_real_loader: bool,
) -> dict:
    """Strict mathematical forecast; not a physical checkpoint acceptance.

    new_trained_core: genuinely new functional trainable parameters beyond
    *existing* 43M architecture. Both legacy + conservative denominator rise.
    activate_old_architecture_training: make existing seed/fixed zero S132
    pathways trainable; legacy denominator already counted all those slots,
    so ONLY conservative denominator may grow. Cap at existing 6.88M gap.

    Zero, fixed, duplicated, masked or untrained coefficients not counted.
    Unpaid physical bytes are never permitted in an ACCEPTANCE certificate.
    """
    check(type(parent_charged_byte_equivalent) in (float,int) and
          0<parent_charged_byte_equivalent<100_000_000 and
          type(grow_unique_trained_params) is int and grow_unique_trained_params>0 and
          type(new_packed_bytes) is int and new_packed_bytes>0 and
          growth_mode in MODES and
          type(newly_functional_trained_params) is int and
          0<=newly_functional_trained_params<=grow_unique_trained_params and
          type(counts_verified_by_real_loader) is bool,
          "invalid or unsupported proposed trained-core growth ledger")
    if growth_mode=="activate_old_architecture_training":
        check(grow_unique_trained_params<=UNTRAINED_OR_FIXED_ARCHITECTURE_GAP,
              "existing architecture training cannot count more than actual latent gap")
    # Forecast denominator uses claimed trainable/functional number for
    # analysis; actual acceptance requires TRUE loader evidence separately.
    new_unique_for_forecast=newly_functional_trained_params
    if growth_mode=="new_trained_core":
        legacy_after=LEGACY_ARCHITECTURE_CORE_PARAMS+new_unique_for_forecast
    else:
        legacy_after=LEGACY_ARCHITECTURE_CORE_PARAMS  # ALREADY COUNTED
    conservative_after=SOURCE_UNIQUE_TRAINED_WEIGHTS+new_unique_for_forecast
    total_bytes=parent_charged_byte_equivalent+new_packed_bytes
    legacy_rate=8*total_bytes/legacy_after
    strict_rate=8*total_bytes/conservative_after
    result={
        "growth_mode":growth_mode,
        "requested_grown_core_parameters":grow_unique_trained_params,
        "verified_functional_learned_new_parameters_FOR_FORECAST":new_unique_for_forecast,
        "claimed_new_encoding_bytes_INCLUDING_metadata":new_packed_bytes,
        "marginal_actual_new_packed_bpp":8*new_packed_bytes/grow_unique_trained_params,
        "legacy_denominator_new":legacy_after,
        "unique_trained_denominator_new":conservative_after,
        "projected_full_charged_bytes":total_bytes,
        "projected_legacy_bpp":legacy_rate,
        "projected_conservative_bpp":strict_rate,
        "projected_legacy_pass_rate_ONLY":legacy_rate<=BPP_MAX,
        "projected_conservative_pass_rate_ONLY":strict_rate<=BPP_MAX,
        "complete_actual_loader_inventory_not_verified":True,
        "future_real_616_replay_required":True,
        "future_matched_CPU_min_ratio":CPU_MIN,
        "future_matched_CPU_target_ratio":CPU_TARGET,
        "future_independent_quality_firstread_required":True,
        "synthetic_growth_or_fixed_zero_cannot_qualify":True,
        "physical_source_growth_checkpoint_NOT_created_here":True,
        "model_accepted":False,
        "activation_allowed":False,
    }
    check(not (result["projected_conservative_pass_rate_ONLY"] and
               newly_functional_trained_params==0),
          "fixed or zero-only parameter expansion can never qualify")
    return result


def minimum_new_genuine_params(*, parent_bytes:float,
                               reference_denominator:int,
                               marginal_bits_per_new_trained_weight:float,
                               new_runtime_metadata_bytes:int=0)->int|None:
    """Minimum MATH-required NEW core params for proposed *marginal* bpp.

    Returns None if the proposed marginal rate is >=.7, in which case
    expansion cannot repair an already-overbudget starting checkpoint.
    This makes denominator-padding tricks fail loudly.
    """
    check(type(reference_denominator) is int and
          reference_denominator in (LEGACY_ARCHITECTURE_CORE_PARAMS,
                                    SOURCE_UNIQUE_TRAINED_WEIGHTS) and
          type(parent_bytes) in (float,int) and parent_bytes>0 and
          type(marginal_bits_per_new_trained_weight) in (float,int) and
          math.isfinite(marginal_bits_per_new_trained_weight) and
          marginal_bits_per_new_trained_weight>=0 and
          type(new_runtime_metadata_bytes) is int and new_runtime_metadata_bytes>=0,
          "invalid packed-core projection")
    debt_bits=8*(parent_bytes+new_runtime_metadata_bytes)-BPP_MAX*reference_denominator
    if debt_bits<=0:
        return 0
    r=float(marginal_bits_per_new_trained_weight)
    if r>=BPP_MAX:
        return None
    # mathematical bound; future physical byte rounding and model metadata
    # can require MORE; this cannot be used as a certification threshold.
    return math.ceil(debt_bits/(BPP_MAX-r))


def audit(s119d_report:Path, original_child:Path)->dict:
    check(s119d_report.name==S119D_REPORT and
          original_child.name==S119D_CHILD,
          "only saved S119d true original12 / old37k physical shadow allowed")
    data=json.loads(s119d_report.read_text(encoding="utf8"))
    check(data.get("stage")==
          "NRA3_S119D_REAL12_SOURCE_A_37087_INWEIGHT_NEURAL_GRAPH_REUSE_GATE0" and
          data.get("S114_original_all12_SynTagRusA_TRAIN_ONLY_source_SHA256")==PARENT_H_SHA and
          data.get("S115_trained_full_53751_source_SHA256")==S115_EXPECTED_SHA256 and
          data.get("inherited_trainable_neural_parameters")==37087 and
          data.get("true_S114_TRAIN_A_rows_seen_without_GOLD")==8 and
          data.get("original_NRA2_eight_mini_areas_exact_on_REAL_original12") is True and
          data.get("original_four_morphology_heads_exact_on_REAL_original12") is True and
          data.get("model_accepted") is False and
          data.get("output_gate")==0 and
          data.get("S118_SynTagRusB_GOLD_never_opened") is True and
          data.get("original_model_and_GreenTunnel_unchanged") is True,
          "S119d immutable 8 mini / all12 previous provenance absent")
    budget=data.get("physical_budget_projection_REQUIRES_LOADER_AUDIT")
    check(isinstance(budget,dict) and
          budget["s194_reference_denominator"]==S194_REFERENCE_DENOM and
          budget["s119a_conservative_trained_core_denominator"]==
          SOURCE_UNIQUE_TRAINED_WEIGHTS and
          data["shared_FP16_checkpoint_file_bytes"]==original_child.stat().st_size and
          data["S119D_repack_full_37087_OLD_neural_child_SHA256"]==sha(original_child),
          "derived physically saved S119d 37k checkpoint SHA or size drift")
    check(budget["projected_legacy_bpp"]>BPP_MAX and
          budget["projected_conservative_bpp"]>BPP_MAX,
          "expected real S119d overbudget source must be audited")
    charged=budget["s194_inferred_charged_parent_byte_equivalent_NOT_physical_file_inventory"] + original_child.stat().st_size
    check(abs(charged-(budget["s194_inferred_charged_parent_byte_equivalent_NOT_physical_file_inventory"]+
             budget["s115_actual_full_FP16_child_file_bytes"]))<1.e-8,
          "missing/replaced historical S194 + S119d physical file")
    # Quantitative no-fantasy scenario grid. Marginal rate MUST be measured
    # with genuine trained weights before anything can be called "success".
    candidates=[]
    for mode in MODES:
        for params in (1_000_000,2_000_000,4_000_000,8_000_000,16_000_000):
            if mode=="activate_old_architecture_training" and params>UNTRAINED_OR_FIXED_ARCHITECTURE_GAP:
                continue
            for r in (.20,.30,.40,.50,.60,.70,.80):
                physical_increment=math.ceil(params*r/8)
                scenario=signed_growth(
                    parent_charged_byte_equivalent=charged,
                    grow_unique_trained_params=params,
                    new_packed_bytes=physical_increment,
                    growth_mode=mode,
                    newly_functional_trained_params=params,
                    counts_verified_by_real_loader=False)
                scenario["assumed_marginal_bpp_NOT_MEASURED"]=r
                candidates.append(scenario)
    limits=[]
    for r in (.20,.30,.40,.50,.55,.60,.65,.70,.80):
        limits.append({
            "assumed_marginal_bpp":r,
            "legacy_min_NEW_ARCH_CORE_PARAMS":
                minimum_new_genuine_params(
                    parent_bytes=charged,
                    reference_denominator=LEGACY_ARCHITECTURE_CORE_PARAMS,
                    marginal_bits_per_new_trained_weight=r),
            "conservative_min_NEW_ARCH_CORE_PARAMS":
                minimum_new_genuine_params(
                    parent_bytes=charged,
                    reference_denominator=SOURCE_UNIQUE_TRAINED_WEIGHTS,
                    marginal_bits_per_new_trained_weight=r),
        })
    return {
        "stage":STAGE,
        "S119d_verified_real_original12_report_sha256":sha(s119d_report),
        "S119d_derived_child_real_FP16_sha256":sha(original_child),
        "S119d_derived_child_real_FP16_bytes":original_child.stat().st_size,
        "S194_previous_historical_charged_byte_equivalent_NOT_FULL_LOADER_INVENTORY":
            budget["s194_inferred_charged_parent_byte_equivalent_NOT_physical_file_inventory"],
        "S194_plus_S119d_full_charged_equivalent_bytes":charged,
        "S194_parent_compressed_source_is_NOT_physically_reinventoried":True,
        "legacy_43261760_includes_old_untrained_S132_growth_channels":True,
        "unique_trainable_36380480_excludes_old_untrained_S132_growth_channels":True,
        "preexisting_fixed_zero_or_seed_derived_architecture_gap":
            UNTRAINED_OR_FIXED_ARCHITECTURE_GAP,
        "S132_old_growth_slots_cannot_be_counted_twice":True,
        "minimum_additional_real_trained_core_params_by_assumed_marginal_bpp":
            limits,
        "forecast_growth_scenarios_NO_REAL_CHECKPOINT_OR_CPU":candidates,
        "no_zero_fixed_or_duplicated_parameter_denominator_padding":True,
        "physical_new_core_training_NOT_PERFORMED":True,
        "original_full616_candidate_NOT_PERFORMED":True,
        "CPU_full_decode_speed_NOT_MEASURED":True,
        "new_independent_understanding_gain_NOT_MEASURED":True,
        "gold_never_imported":True,
        "old_616_source_report_only":True,
        "S118_B_heldout_still_unopened":True,
        "model_accepted":False,
        "activation_allowed":False,
        "output_gate":0,
        "decision":"CORE_GROWTH_ALLOWED_FOR_RESEARCH_ONLY_IF_GENUINELY_LEARNED_AND_PHYSICALLY_CHARGED",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--s119d-report",type=Path,required=True)
    p.add_argument("--s119d-child",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    check(not args.out.exists(),"no overwrite/reuse of staged growth result")
    result=audit(args.s119d_report,args.s119d_child)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("NRA3_S119E_REAL_PARENT_CORE_GROWTH_MATH "+
          json.dumps({
              "real_derived_child_bytes":result["S119d_derived_child_real_FP16_bytes"],
              "parent_equivalent_bytes":result["S194_plus_S119d_full_charged_equivalent_bytes"],
              "required":result["minimum_additional_real_trained_core_params_by_assumed_marginal_bpp"],
              "scenarios":len(result["forecast_growth_scenarios_NO_REAL_CHECKPOINT_OR_CPU"]),
              "accepted":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
