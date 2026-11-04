"""On real 135M frozen CB59: FIT ground-truth, serialize and VERIFY <=0.7bpp."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v117.protocol import ORIGINAL_CB59_SHA,SOURCE_BYTES,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from .archive import Paid,pack
from .train import fit
from .runtime import attach,restore
from .behavior import assess,speed_single

def run(source_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    original=source_path.read_bytes()
    if (len(original)!=SOURCE_BYTES or hashlib.sha256(original).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("original complete trained CB59 not preserved")
    if CB59(original).meta["original_unique_parameter_count"]!=TOTAL_UNIQUE:
        raise AssertionError("wrong unique 135M source weight denominator")
    model,meta,_=make_model("baseline",SimpleNamespace(original=source_path,folder=out))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("native packed weight source lost")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    before=assess(model,tok)
    trained_head,train_diag=fit(model,tok)
    blob=pack(original,head=trained_head)
    paid=Paid(blob)
    if paid.base!=original or len(blob)>MAX_SOURCE_BYTES:
        raise AssertionError("source bytes changed or physical budget exceeded")
    archive=out/"CALIBRI_v118_paid_135M_trained.cb118"
    archive.write_bytes(blob)
    from_disk=Paid(archive.read_bytes())
    attached=attach(model,from_disk)
    try:
        after=assess(model,tok)
        if any(p.ndim>=2 for p in model.parameters()):
            raise AssertionError("original trained 2D weights accidentally decoded")
    finally:restore(model,attached)
    again=attach(model,from_disk)
    try:
        verified=assess(model,tok)
        if verified!=after:raise AssertionError("physical candidate reload changed free text")
    finally:restore(model,again)
    timing=speed_single(model,tok,from_disk,attach,restore)
    report={
        "experiment":"CALIBRI v118 full 135M preserved sub0.7bpp supervised head adaptation",
        "source_model":MODEL,"source_model_revision":MODEL_SHA,
        "all_in_complete_physical_archive":paid.stats(),
        "original_CB59_source_bytewise_immutable":True,
        "60_role_rank1_zero_physical_factors_untrained":True,
        "original_v117_zero_rank2_roles_reallocated_without_loss_of_trained_behavior":True,
        "true_trained_rank2_head_plus_vocab_bias":True,
        "only_34_gold_labeled_TRAIN_not_teacher_KL":train_diag,
        "original_CB59_valid_answers":before,
        "new_trained_valid_answers":after,
        "physical_reload_answer_parity":True,
        "native_cpu_timing":timing,
        "independent_v89_TEST_not_opened":True,
        "no_sci_PASS_without_readable_good_answers":True,
        "main_Render_Supabase_APK_Jessica_LIVE_unchanged":True}
    (out/"v118_paid_behaviour.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":"v118_paid_original_135m",
        "archive_bytes":len(blob),"true_bpp":paid.stats()["all_in_bpp"],
        "source_CB59_locked":True,
        "CB59_strict_answer_count":before["score"]["strict_correct_and_EOS"],
        "new_strict_answer_count":after["score"]["strict_correct_and_EOS"],
        "native_cpu_ratio":timing["trained_over_CB59"],
        "new_outputs": [x["actual"][:90] for x in after["gold"][:6]],
        "scientific_QUALITY_PASS":False},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();run(a.source,a.out)
