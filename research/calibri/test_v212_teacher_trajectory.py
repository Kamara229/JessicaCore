"""v212 regression: only frozen TRAIN, FP32 self-generated full-vocab paths."""
import unittest
from types import SimpleNamespace

import torch
from v196 import protocol as p
from v212.teacher_trajectory import (
    TRAJECTORY_STEPS, TEMPERATURE, freeze_fp32_train_trajectories,
    trajectory_distillation_loss,
)
from v212.run import PILOT_STEPS, SOURCE_STEP_OFFSET, balanced_domain_index

class SimpleVocabModel(torch.nn.Module):
    def __init__(self, token=3):
        super().__init__()
        logits=torch.zeros(p.VOCAB_SIZE,dtype=torch.float32)
        logits[token]=4.
        self.table=torch.nn.Parameter(logits)
    def forward(self,input_ids,use_cache=False):
        return SimpleNamespace(logits=self.table[None,None,:].expand(
            input_ids.shape[0],input_ids.shape[1],p.VOCAB_SIZE))

def mock_frozen_samples():
    train=[]
    for idx in p.PUBLIC_TRAIN_INDICES:
        if idx<32:
            train.append({"kind":"wiki", "answer_start":None,
                "ids":torch.arange(64,dtype=torch.long)})
        else:
            train.append({"kind":"supervised_chat", "answer_start":3,
                "ids":torch.tensor([5,6,7,8,9],dtype=torch.long)})
    return train

class TestV212TeacherTrajectory(unittest.TestCase):
    def test_full_original_20_train_and_actual_teacher_greedy(self):
        teacher=SimpleVocabModel(token=3)
        frozen=freeze_fp32_train_trajectories(
            teacher,mock_frozen_samples(),eos_id=9,steps=TRAJECTORY_STEPS)
        self.assertEqual(len(frozen),20)
        self.assertEqual([r["source_public_train_index"] for r in frozen],
                         list(p.PUBLIC_TRAIN_INDICES))
        for r in frozen:
            self.assertEqual(r["split"],"TRAIN")
            self.assertTrue(r["FP32_teacher_self_generated_not_gold_answer"])
            self.assertEqual(r["teacher_greedy_next_ids"].tolist(),[3]*TRAJECTORY_STEPS)
            self.assertEqual(r["teacher_full_vocab_probs_fp16"].shape,
                             (TRAJECTORY_STEPS,p.VOCAB_SIZE))
            self.assertEqual(len(r["original_prefix_token_sha256"]),64)
        self.assertEqual(len(frozen[0]["prefix"]),24)
        self.assertEqual(len(frozen[-1]["prefix"]),3)

    def test_train_trajectory_guides_paid_student_through_gradient(self):
        teacher=SimpleVocabModel(token=3)
        student=SimpleVocabModel(token=4)
        frozen=freeze_fp32_train_trajectories(
            teacher,mock_frozen_samples(),eos_id=9,steps=3)
        loss,record=trajectory_distillation_loss(student,frozen[0])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(record["complete_teacher_49152_vocab"])
        loss.backward()
        self.assertTrue(torch.isfinite(student.table.grad).all())
        self.assertLess(student.table.grad[3].item(),0.)

    def test_prevent_validation_leakage_and_wrong_temperature(self):
        teacher=SimpleVocabModel()
        student=SimpleVocabModel(token=4)
        frozen=freeze_fp32_train_trajectories(
            teacher,mock_frozen_samples(),eos_id=9,steps=2)
        v=dict(frozen[0]);v["split"]="VALID"
        with self.assertRaises(AssertionError):
            trajectory_distillation_loss(student,v)
        with self.assertRaises(AssertionError):
            trajectory_distillation_loss(student,frozen[0],temperature=TEMPERATURE+1)
        with self.assertRaises(AssertionError):
            freeze_fp32_train_trajectories(
                teacher,mock_frozen_samples()[:19],eos_id=9)

    def test_early_teacher_eos_is_handled_without_synthetic_followup(self):
        teacher=SimpleVocabModel(token=9)
        frozen=freeze_fp32_train_trajectories(
            teacher,mock_frozen_samples(),eos_id=9,steps=4)
        self.assertTrue(all(len(x["teacher_greedy_next_ids"])==1 for x in frozen))
        student=SimpleVocabModel(token=4)
        loss,_=trajectory_distillation_loss(student,frozen[0])
        self.assertTrue(torch.isfinite(loss))

    def test_24_unique_new_paid_operators_and_train_pair_coverage(self):
        self.assertEqual(SOURCE_STEP_OFFSET,48)
        self.assertEqual(PILOT_STEPS,24)
        old=set(range(48))
        new=list(range(SOURCE_STEP_OFFSET,SOURCE_STEP_OFFSET+PILOT_STEPS))
        self.assertTrue(old.isdisjoint(new))
        self.assertEqual(len(new),len(set(new)))
        self.assertEqual(sorted(balanced_domain_index(x) for x in range(PILOT_STEPS)),
                         list(range(24)))

if __name__=="__main__":
    unittest.main()
