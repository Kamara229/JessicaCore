"""Z98: frozen Z96 transfer to 36 existing non-definition Z87 TRAIN rows.
These rows were not used in the Z96 micro-fit. They are not fresh DEV,
and were used in other historical branches. Never use spent Z89 prompts.
"""
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import prepare as prepare_all,PARENT_SHA,GT_SHA
JOINT_SHA="292edde21c3128bdddb2e510c29db0f7fed452621ae9504f2bebf38bea83f917"
def verify_z96(r):
    check(r["stage"]=="S119Q2I8Z96_GPT_MICRO_CONVERGENCE" and r["Z73_revived_parent_SHA256"]==PARENT_SHA
          and r["GT12_SHA256"]==GT_SHA and r["training"]["Adam_updates"]==600 and r["training"]["cached_training_positions"]==30600
          and r["own_metrics"]["AFTER"]["complete_correct"]==12 and r["reference_metrics"]["AFTER"]["top1"]==51
          and r["saved_research_only"]["joint"]["sha256"]==JOINT_SHA and r["saved_research_only"]["GT12"]["sha256"]==GT_SHA
          and r["FP32_calls"]==0 and r["output_gate"]==0 and r["candidate_accepted"] is False
          and r["diagnostic_only_never_promote"],"exact complete Z96 diagnostic only")
def prepare(tok):
    _,_,new,authored=prepare_all(tok)
    train=[r for r in new if authored[r["id"]]["family"]=="definition"]
    transfer=[r for r in new if authored[r["id"]]["family"]!="definition"]
    check(len(train)==12 and len(transfer)==36 and sum(len(r["answer_native_with_eos"]) for r in transfer)==153
          and not {r["id"] for r in train}.intersection(r["id"] for r in transfer),"exact 36 distinct contexts not used in Z96 fit")
    check({authored[r["id"]]["completion"] for r in train}=={authored[r["id"]]["completion"] for r in transfer},"same twelve target words")
    return transfer,authored
def own_metrics(records):
    check(len(records)==36 and len({r["id"] for r in records})==36,"complete context diagnostic")
    def sub(rs):
        return {"n":len(rs),**{k:sum(r[k] for r in rs) for k in ("complete_correct","text_correct","EOS","valid_UTF8")},
            "first_two_correct":sum(r["prefix_match"]>=2 for r in rs),"forwards":sum(len(r["generated_ids"]) for r in rs)}
    groups={f:sub([r for r in records if r["family"]==f]) for f in ("cloze","contrast","plural")}
    check(all(v["n"]==12 for v in groups.values()),"balanced non-definition contexts")
    return {**sub(records),"by_family":groups}
