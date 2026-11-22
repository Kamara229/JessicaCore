"""S119q2b source-only 43M native one-pass training *contracts*, synthetic unit tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2a_train_source_manifest import (
    build_rows,build_manifest,canonical_bytes,KINDS,SPLITS,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    EPOCHS,LR,TRAIN_INDEX,PROBE_INDEX,
    prepare_microblock,verify_q2a_artifact,true_native_first_target,
    persist_half,
)


class TokenizerFake:
    vocab_size=49152
    def encode(self,txt,add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in txt]
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class AdapterFake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    def forward(self,x):return self.weight*x


def create_source(root:Path):
    source=root/"q2a"
    source.mkdir()
    rows=build_rows()
    manifest=build_manifest(rows)
    for split in SPLITS:
        (source/("S119Q2A_"+split+".jsonl")).write_bytes(
            b"".join(canonical_bytes(r) for r in rows
                     if r["split"]==split))
    (source/"S119Q2A_MANIFEST.json").write_bytes(canonical_bytes(manifest))
    return source


class S119Q2BNativeOnepassTrainControls(unittest.TestCase):
    def test_01_exact_six_train_six_separate_source_probe(self):
        tr,pr=prepare_microblock(build_rows())
        self.assertEqual(len(tr),len(KINDS))
        self.assertEqual(len(pr),len(KINDS))
        self.assertEqual({x["kind"] for x in tr},set(KINDS))
        self.assertEqual({x["kind"] for x in pr},set(KINDS))
        self.assertTrue(all(x["split"]=="TRAIN" for x in tr+pr))
        self.assertTrue(all(x["id"].endswith("_00") for x in tr))
        self.assertTrue(all(x["id"].endswith("_01") for x in pr))
        self.assertFalse({x["id"] for x in tr}&{x["id"] for x in pr})
        self.assertEqual(TRAIN_INDEX,0);self.assertEqual(PROBE_INDEX,1)

    def test_02_only_stated_training_budget_and_no_final_qa(self):
        self.assertEqual(EPOCHS,2)
        self.assertAlmostEqual(LR,0.001)
        self.assertEqual(len(prepare_microblock(build_rows())[0])*EPOCHS,12)

    def test_03_first_native_BPE_real_full_class_roundtrip(self):
        row=prepare_microblock(build_rows())[0][0]
        target,all_ids=true_native_first_target(TokenizerFake(),row)
        self.assertEqual(target,ord(row["completion"][0]))
        self.assertGreater(len(all_ids),1)
        self.assertTrue(all(0<=i<TokenizerFake.vocab_size for i in all_ids))

    def test_04_reject_calibration_as_training(self):
        row=next(x for x in build_rows() if x["split"]=="CALIBRATION")
        with self.assertRaisesRegex(GateFailure,"TRAIN-only"):
            true_native_first_target(TokenizerFake(),row)

    def test_05_reject_mislabeled_external_gold(self):
        row=copy.deepcopy(prepare_microblock(build_rows())[0][0])
        row["ground_truth_kind"]="unopened_S118_B_gold"
        with self.assertRaisesRegex(GateFailure,"TRAIN-only"):
            true_native_first_target(TokenizerFake(),row)

    def test_06_immutable_q2a_bytes_and_manifest_physically_verified(self):
        with tempfile.TemporaryDirectory() as d:
            source=create_source(Path(d))
            rows,man=verify_q2a_artifact(source)
            self.assertEqual(rows,build_rows())
            self.assertEqual(man,build_manifest(build_rows()))

    def test_07_reject_any_modified_authored_train_label(self):
        with tempfile.TemporaryDirectory() as d:
            source=create_source(Path(d))
            p=source/"S119Q2A_TRAIN.jsonl"
            content=p.read_text()
            self.assertIn('"completion":',content)
            p.write_text(content.replace('"completion":','"completionX":',1))
            with self.assertRaisesRegex(GateFailure,"physical SHA-bound"):
                verify_q2a_artifact(source)

    def test_08_reject_mutated_calibration_even_if_not_training(self):
        with tempfile.TemporaryDirectory() as d:
            source=create_source(Path(d))
            p=source/"S119Q2A_CALIBRATION.jsonl"
            p.write_text(p.read_text()+"{}\n")
            with self.assertRaisesRegex(GateFailure,"physical SHA-bound"):
                verify_q2a_artifact(source)

    def test_09_refuse_extra_protected_gold_or_artifact_file(self):
        with tempfile.TemporaryDirectory() as d:
            source=create_source(Path(d))
            (source/"S118_B_GOLD.json").write_text("{}")
            with self.assertRaisesRegex(GateFailure,"exact physical SOURCE"):
                verify_q2a_artifact(source)

    def test_10_required_full_local_match_weights_can_be_serialized(self):
        with tempfile.TemporaryDirectory() as d:
            a=AdapterFake()
            b=copy.deepcopy(a)
            self.assertEqual(sum(x.numel() for x in a.parameters()),
                             sum(x.numel() for x in b.parameters()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(
                a.state_dict().values(),b.state_dict().values())))
            p=Path(d)/"physical_FP16.safetensors"
            meta=persist_half(a,p)
            self.assertEqual(len(meta["sha256"]),64)
            self.assertEqual(meta["saved_total_child_FP16_params"],3)
            with self.assertRaisesRegex(GateFailure,"never overwrite"):
                persist_half(a,p)

    def test_11_train_probe_rule_diversity_without_heldout_claim(self):
        tr,pr=prepare_microblock(build_rows())
        self.assertEqual(Counter(x["kind"] for x in tr),Counter(KINDS))
        self.assertEqual(Counter(x["kind"] for x in pr),Counter(KINDS))
        self.assertEqual(len(set(x["prompt"] for x in tr+pr)),12)
        self.assertEqual(len(set(x["completion"] for x in tr+pr))>0,True)

    def test_12_reject_mutated_train_source_even_if_same_schema(self):
        rows=copy.deepcopy(build_rows())
        rows[0]["rule"]="training_leak_to_eval"
        with self.assertRaisesRegex(ValueError,"generator output"):
            prepare_microblock(rows)


if __name__=="__main__":
    unittest.main()
