"""Stage-2: authentic 616-question native CPU J7 with passive NRA2 read port.

Reuses the *unmodified* S194/S190/S187/S168 generation path. Load a separately
TRAINED graph NRA (pinned independent FP16), attach a single read-only last
decoder-layer hook to original 43M core, and verify exact immutable generation
SHA for 456+128+32. No core/NRA learning, activation or output changes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from calibri2.s194.actual_cpu_j7_bridge import real_j7_baseline, output_digest
from nra2_bridge.passive_final_layer import NRA2PassiveFinalLayer


def main():
    if len(sys.argv)<15:
        raise SystemExit("NRA2 bridge requires full frozen S194 original CPU J7 CLI")
    args=list(sys.argv)
    out=Path(args[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Immutable NRA2 J7 output directory required")

    pinned_path=Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json")
    standalone_path=Path("inputs/nra2/NRA2_STANDALONE_REPORT.json")
    checkpoint=Path("inputs/nra2/NRA2_STANDALONE_RELATIONAL_FP16.pt")
    if not (pinned_path.is_file() and standalone_path.is_file() and checkpoint.is_file()):
        raise AssertionError("Pinned S194 or successful independent NRA2 sources missing")
    pinned=json.loads(pinned_path.read_text())
    prior=json.loads(standalone_path.read_text())
    prior_cert=pinned["J7Retention584_frozen_baseline"]
    if (pinned.get("stage")!="S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or not pinned.get("all_616_exact_real_CPU_generated_tokens_captured")
            or pinned.get("candidate_weight_update_executed") is not False
            or prior.get("stage")!="NRA2_STANDALONE_RELATIONAL_V0"
            or prior.get("candidate_status")!="EXPLORATORY_STANDALONE_PASS_NOT_INTEGRATION"
            or prior.get("calibri_2_integrated") is not False
            or prior["architecture"]["trainable_parameters"]!=2241
            or prior["novel_graph_size_five_hop_dev"]["correct"]!=1024):
        raise AssertionError("Prior S194/NRA2 provenance drift")

    # Exact original s112 loader is patched ONLY in this test process.
    # s168 captures original decoder on import, so patch before lazy s190.
    from calibri2.s112 import prompt_conditioned_first_token_mps as s112
    previous_load=s112.load_cpu_original
    attached=[]
    def load_with_nra2_port(*a,**kw):
        cpu=previous_load(*a,**kw)
        if attached:
            raise AssertionError("NRA2 must bind to exactly ONE Calibri CPU core")
        bridge=NRA2PassiveFinalLayer(checkpoint,
                            prior["fp16_checkpoint_sha256"]).install(cpu[0])
        attached.append(bridge)
        return cpu
    s112.load_cpu_original=load_with_nra2_port
    try:
        old,new,expected128,old_gold,probes,probe_gold,eos,source=real_j7_baseline(
            args,old_s87=Path(args[-2]))
    finally:
        s112.load_cpu_original=previous_load
        for bridge in attached:
            bridge.uninstall()

    if len(attached)!=1 or attached[0].forward_calls<1:
        raise AssertionError("NRA2 last-layer port did not observe any real CPU frames")
    if source["original_real_CPU_FULL_EOS_456"]!=456 or source["new128_real_CPU_FULL_EOS"]!=128:
        raise AssertionError("NRA2 changed old/new real CPU J7 answers")
    protected=dict(old)
    if set(protected).intersection(new) or (set(protected)|set(new)).intersection(probes):
        raise AssertionError("NRA2 J7 identity overlap")
    protected.update(new)
    if (output_digest(protected)!=prior_cert["protected_584_generation_digest"]
            or output_digest(probes)!=prior_cert["train32_generation_digest"]):
        raise AssertionError("NRA2 passive port changed physical S194 token IDs")
    if (sum(old[k]==old_gold[k] for k in old)!=456
            or sum(new[k]==expected128[k] for k in new)!=128
            or sum(probes[k]==probe_gold[k] for k in probes)!=0
            or sum(bool(probes[k]) and probes[k][0]==probe_gold[k][0] for k in probes)!=16):
        raise AssertionError("NRA2 passive CPU outputs drifted from S194")
    if attached[0].report["connected_decoder_layers"]!=1:
        raise AssertionError("NRA2 unexpectedly connected to all 12 layers")
    out.mkdir(parents=True,exist_ok=True)
    report={
       **attached[0].report,
       "original_core": "frozen S194 actual 43M CPU",
       "independently_trained_NRA_loaded":True,
       "original_456_full_EOS":456,"new128_full_EOS":128,
       "novel32_full_EOS":0,"novel32_first":16,
       "exact_584plus32_real_generated_token_ID_digest_matches_S194":True,
       "protected_generation_sha256":output_digest(protected),
       "new32_generation_sha256":output_digest(probes),
       "J7_baseline_full_retention":True,
       "candidate_enabled":False,
       "CPU_speed_candidate_measured":False,
       "all_in_bpp_candidate_measured":False,
       "NRA2_hidden_to_graph_adapter_trained":False,
       "NRA2_output_influences_generation":False,
       "Jessica_LIVE_Render_Supabase_APK_untouched":True,
       "model_accepted":False,
    }
    (out/"NRA2_REAL_43M_PASSIVE_616_J7.json").write_text(
         json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
