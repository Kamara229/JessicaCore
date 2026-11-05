"""CALIBRI v171: source inference purity, no per-question native recompiles."""
import inspect
import unittest
from unittest.mock import patch
from v169.sources import all_cases,digest
from v169.run import gen
from v169.protocol import ARMS

class PersistentPaidRuntime(unittest.TestCase):
    def test_all_576_real_arm_calls_are_paired_without_toggling_models(self):
        cases=all_cases()
        model_objects={a:object() for a in ARMS}
        calls=[]
        def fake_generate(model,tok,prompt,stop_at_eos=False):
            calls.append((id(model),prompt))
            return {"ids":[1,2],"text":"stub",
                    "prompt_SHA":digest(prompt),
                    "elapsed_seconds":0.01,
                    "prefill_seconds":0.003,
                    "decode_seconds":0.007,
                    "eos_generated":False}
        with patch("v169.run.generate_one",side_effect=fake_generate),\
             patch("v169.run.build_chat",side_effect=lambda tok,q:"CHAT::"+q):
            scored=gen(model_objects,object(),cases)
        self.assertEqual(len(calls),576)
        for arm in ARMS:
            self.assertEqual(len(scored[arm]),192)
            self.assertTrue(all(x["verbatim_text"]=="stub" for x in scored[arm]))
            self.assertEqual([x["prompt_sha"] for x in scored[arm]],
                             [c["prompt_sha"] for c in cases])
        # All three distinct original model bindings were actually evaluated.
        self.assertEqual({id(model_objects[a]) for a in ARMS},
                         {id(m) for m in model_objects.values()})

    def test_gen_never_compiles_or_temporarily_swaps_paid_layers(self):
        code=inspect.getsource(gen)
        self.assertNotIn("install_paid_block(",code)
        self.assertNotIn("compile_lut(",code)
        self.assertNotIn("make_model(",code)
        self.assertIn("models[arm]",code)

if __name__=="__main__":
    unittest.main()
