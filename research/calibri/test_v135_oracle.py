import unittest
from types import SimpleNamespace
import torch
from torch import nn
from v135.arms import ORACLE_ARMS,BASE_ARMS,PAID_ARMS,oracle_modules,set_paid
from v135.scoring import summarize
from v134.protocol import corpus


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(32,8)
        self.lm_head=nn.Linear(8,32,bias=False)


class Bridge:
    def __init__(self):
        self.blocks={f"model.layers.{i}.{role}":SimpleNamespace(gain=1.)
                     for i in range(30) for role in ("self_attn.o_proj","mlp.down_proj")}
    def control(self,policy,question):
        for layer in self.blocks.values():layer.gain=1.


class TestOracleIsolation(unittest.TestCase):
    def test_real_teacher_borrow_is_restored_on_exit(self):
        student,teacher=Toy(),Toy()
        source_embedding=student.model.embed_tokens
        source_head=student.lm_head
        with oracle_modules(student,teacher,input=True,head=True):
            self.assertIs(student.model.embed_tokens,teacher.model.embed_tokens)
            self.assertIs(student.lm_head,teacher.lm_head)
        self.assertIs(student.model.embed_tokens,source_embedding)
        self.assertIs(student.lm_head,source_head)

    def test_restores_even_after_exception(self):
        student,teacher=Toy(),Toy()
        old_embedding=student.model.embed_tokens
        with self.assertRaises(RuntimeError):
            with oracle_modules(student,teacher,input=True):
                raise RuntimeError("probe failed")
        self.assertIs(student.model.embed_tokens,old_embedding)

    def test_paid_first_five_layers_disabled_only_under_drop(self):
        bridge=Bridge()
        self.assertEqual(len(corpus()),16)
        self.assertEqual(len(BASE_ARMS),4)
        self.assertEqual(len(PAID_ARMS),4)
        self.assertEqual(len(ORACLE_ARMS),8)
        for arm in PAID_ARMS:
            self.assertTrue(set_paid(bridge,arm))
            for path,layer in bridge.blocks.items():
                n=int(path.split(".")[2])
                self.assertEqual(layer.gain,0. if n<5 and arm.startswith("v119_no_0to4") else 1.)

    def test_report_rejects_incomplete(self):
        with self.assertRaises(ValueError):
            summarize({"CB59":[]})


if __name__=="__main__":
    unittest.main()
