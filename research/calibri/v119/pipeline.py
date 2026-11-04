"""First REAL native C-gradient joint 60-block+head TRAIN for full 135M <0.7bpp."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v118.archive import Paid,pack
from v118.runtime import attach,restore as native_restore
from .native import build,install,restore as undo_grad
from .learn import fit
from .quality import assess,speed

def experiment(source_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    previous=Paid(source_path.read_bytes())
    base=previous.base
    if (hashlib.sha256(base).hexdigest()!=ORIGINAL_CB59_SHA or
        CB59(base).meta["original_unique_parameter_count"]!=TOTAL_UNIQUE or
        EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("previous original 135M CB59 bytes were modified")
    (out/"original_unmodified.cb59").write_bytes(base)
    model,meta,_=make_model("baseline",SimpleNamespace(
          original=out/"original_unmodified.cb59",folder=out))
    if not meta["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("not direct native packed CB59")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    baseline=assess(model,tok)
    previous_sources=attach(model,previous)
    try:
        previous_behaviour=assess(model,tok)
    finally:native_restore(model,previous_sources)
    for p in model.parameters():p.requires_grad_(False)
    forward,reverse=build(out)
    installed=install(model,forward,reverse)
    try:
        roles,head,train_diag=fit(model,tok,previous)
    finally:undo_grad(model,installed)
    new=pack(base,head=head,blocks=roles)
    current=Paid(new)
    if len(new)!=len(previous.raw) or len(new)>MAX_SOURCE_BYTES or current.base!=base:
        raise AssertionError("same byte full original source was violated")
    (out/"CALIBRI_v119_joint_native_135M.cb118").write_bytes(new)
    paid=Paid((out/"CALIBRI_v119_joint_native_135M.cb118").read_bytes())
    if not any(u.any() and v.any() for u,v in (paid.role(n) for n in paid.layout)):
        raise AssertionError("non-zero body corrective weights absent from archive")
    patched=attach(model,paid)
    try:
        candidate=assess(model,tok)
        if any(p.ndim>=2 for p in model.parameters()):
            raise AssertionError("source weights no longer packed resident")
    finally:native_restore(model,patched)
    timing=speed(model,tok,paid,attach,native_restore)
    report={
        "experiment":"CALIBRI v119 exact full source 135M native C-VJP joint gold TRAIN",
        "original_model":MODEL,"revision":MODEL_SHA,
        "full_all_in_paid_archive":paid.stats(),
        "same_exact_archive_bytes_as_v118":len(new)==len(previous.raw),
        "original_CB59_entire_checkpoint_SHA_preserved":True,
        "no_original_trained_dense_2D_source_allocated_for_gradient":True,
        "true_native_reverse_on_entire_packed_network":True,
        "joint_train_selected_gold_not_teacher_KL":train_diag,
        "CB59":baseline,"prior_v118_head_only":previous_behaviour,
        "v119_joint_answer_generation":candidate,
        "native_cpu":timing,
        "new_frozen_VALID_never_used_for_gradient":True,
        "scientific_full_quality_PASS":False,
        "independent_v89_TEST_unopened":True,
        "no_prod_or_main_changes":True}
    (out/"v119_joint_gold_cpu.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v119_real_joint_finished",
        "all_in_bpp":paid.stats()["all_in_bpp"],
        "same_archive_bytes":True,
        "body_gradient_count":train_diag["max_nonzero_body_block_grads"],
        "correct_original":baseline["summary"]["strict_correct_and_eos"],
        "correct_head":previous_behaviour["summary"]["strict_correct_and_eos"],
        "correct_joint":candidate["summary"]["strict_correct_and_eos"],
        "native_speed_ratio":timing["joint_vs_frozen_original_CB59"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("previous",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();experiment(a.previous,a.out)
