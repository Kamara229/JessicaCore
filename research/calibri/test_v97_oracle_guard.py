"""Fast observable checks that FP32 oracle never persists in native source."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from direct_runtime_v59 import PackedLinear,PackedEmbedding
from v97.oracle import temporary_oracle_modules
from v97.spec import TRIALS,BOOLQ_RANGE

class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Module()
        self.a.linear=PackedLinear(SimpleNamespace(shape=(4,4)),None)
        self.a.embed=PackedEmbedding(SimpleNamespace(shape=(9,4)),None)

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Module()
        self.a.linear=nn.Linear(4,4)
        self.a.embed=nn.Embedding(9,4)

class Tests(unittest.TestCase):
    def test_context_manager_restores_original_on_success(self):
        native,teacher=Example(),Teacher()
        old=native.a.linear
        with temporary_oracle_modules(native,teacher,("a.linear",)):
            self.assertIs(native.a.linear,teacher.a.linear)
        self.assertIs(native.a.linear,old)

    def test_context_manager_restores_original_on_exception(self):
        native,teacher=Example(),Teacher()
        emb,lin=native.a.embed,native.a.linear
        with self.assertRaisesRegex(ValueError,"fail-after-swap"):
            with temporary_oracle_modules(native,teacher,("a.embed","a.linear")):
                self.assertIs(native.a.embed,teacher.a.embed)
                self.assertIs(native.a.linear,teacher.a.linear)
                raise ValueError("fail-after-swap")
        self.assertIs(native.a.embed,emb)
        self.assertIs(native.a.linear,lin)

    def test_no_partial_replace_when_second_module_invalid(self):
        native,teacher=Example(),Teacher()
        old=native.a.linear
        teacher.a.embed=nn.Linear(4,4) # invalid FP32 embedding type
        with self.assertRaises(AssertionError):
            with temporary_oracle_modules(native,teacher,("a.linear","a.embed")):pass
        self.assertIs(native.a.linear,old)

    def test_no_unpaid_operator_or_conflicting_probes(self):
        native,teacher=Example(),Teacher()
        with self.assertRaises(AssertionError):
            with temporary_oracle_modules(native,teacher,("a.linear","a.linear")):pass
        with self.assertRaises(AssertionError):
            with temporary_oracle_modules(native,teacher,("a.missing",)):pass

    def test_preregistered_new_train_only_questions(self):
        self.assertEqual(BOOLQ_RANGE[1]-BOOLQ_RANGE[0],16)
        self.assertGreaterEqual(BOOLQ_RANGE[0],8414)
        self.assertEqual(len(TRIALS),6)
        self.assertIn("lm_head",[m for _,paths in TRIALS for m in paths])

if __name__=="__main__":unittest.main()
