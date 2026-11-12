"""S146 pure-CPU retention and physical accounting fail-closed contracts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calibri2.s146.first_real_neural_skill import retention,candidate_bytes_and_bpp
from calibri2.s145.neural_skill_blocks import LayerwiseNeuralSkillBank,NeuralSkillSpec


def fixture():
    old={f"old/{i}":{"expected":[i,99]} for i in range(5)}
    parent_old={k:list(row["expected"]) for k,row in old.items()}
    novel={}
    for i in range(64):
        for side in ("left","right"):
            name=f"new/{i}/{side}"
            novel[name]={
                "id":name,"pair_id":f"pair/{i}","side":side,
                "expected":[i,1 if side=="left" else 2,99],
            }
    parent_new={k:list(row["expected"]) for k,row in novel.items()}
    return old,parent_old,novel,parent_new


class RetentionContracts(unittest.TestCase):
    def test_exact_token_and_eos_identity_all_retained(self):
        old,original,new,novel=fixture()
        r=retention(original,original,old,novel,novel,new)
        self.assertEqual(r["old_full_before"],5)
        self.assertEqual(r["old_full_after"],5)
        self.assertEqual(r["new_full_after"],128)
        self.assertEqual(r["new_pairs_both_after"],64)
        self.assertFalse(any(r[k] for k in ("lost_old_full_ids","lost_old_first_ids",
                      "lost_new_full_ids","lost_new_first_ids")))

    def test_old_first_token_loss_cannot_hide_behind_full_count(self):
        old,parent,new,parent_new=fixture()
        candidate={k:list(v) for k,v in parent.items()}
        candidate["old/0"]=[123,99]
        candidate["old/1"]=[1,77]
        result=retention(parent,candidate,old,parent_new,parent_new,new)
        self.assertEqual(result["lost_old_full_ids"],["old/0","old/1"])
        self.assertEqual(result["lost_old_first_ids"],["old/0"])

    def test_new_first_token_loss_is_explicit(self):
        old,parent,new,pnew=fixture()
        candidate={k:list(v) for k,v in pnew.items()}
        candidate["new/0/left"]=[999,2,99]
        r=retention(parent,parent,old,pnew,candidate,new)
        self.assertEqual(r["lost_new_full_ids"],["new/0/left"])
        self.assertEqual(r["lost_new_first_ids"],["new/0/left"])
        self.assertEqual(r["new_pairs_both_before"],64)
        self.assertEqual(r["new_pairs_both_after"],63)

    def test_eos_only_loss_is_protected_without_false_first_loss(self):
        old,parent,new,pnew=fixture()
        candidate={k:list(v) for k,v in pnew.items()}
        candidate["new/63/right"]=[63,2,777]
        r=retention(parent,parent,old,pnew,candidate,new)
        self.assertEqual(r["lost_new_full_ids"],["new/63/right"])
        self.assertEqual(r["lost_new_first_ids"],[])

    def test_real_geometry_all_bytes_accounted(self):
        bank=LayerwiseNeuralSkillBank(320,12)
        bank.add_skill(NeuralSkillSpec("binding_v1",(3,9),8))
        class Q2:
            def new_parameter_count(self):return 2*320*4+24
        class Expansion:
            def architectural_new_parameter_count(self):return 6_881_280
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"skill.safetensors"
            manifest=Path(td)/"manifest.json"
            file.write_bytes(b"0"*22000)
            manifest.write_bytes(b"{}")
            info=candidate_bytes_and_bpp(
                {"physical_bpp":0.6568244742},file,manifest,Q2(),Expansion(),bank)
        self.assertEqual(info["new_neural_skill_params"],10882)
        self.assertEqual(info["new_total_physical_bytes"],22002)
        self.assertGreater(info["all_in_physical_bpp"],0.6568244742)
        self.assertLess(info["all_in_physical_bpp"],.7)


if __name__=="__main__":
    unittest.main()
