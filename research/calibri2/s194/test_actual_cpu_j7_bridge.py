"""S194 schema/identity tests: simulated data only, real J7 runs separate CI."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from calibri2.s194.actual_cpu_j7_bridge import (
    make_real_584_certificate,output_digest,actual_fp16_checkpoint_sha,
    one_step_with_actual_physical_cpu_gate,
)

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def data():
    old={sha(f"old{i}"):[3,4,0] for i in range(456)}
    new={sha(f"new{i}"):[5,6,0] for i in range(128)}
    gold={sha(f"probe{i}"):[7,8,0] for i in range(32)}
    probes={k:([7,9,0] if i%2==0 else [10,8,0])
            for i,k in enumerate(gold)}
    return old,new,gold,probes

def cert():
    old,new,gold,probes=data()
    return make_real_584_certificate(
        old_pred=old,old_target=old,new_pred=new,new_target=new,
        probe_pred=probes,probe_target=gold,eos_id=0,
        source_bpp=.699,speed_ratio=.95,checkpoint_sha256="a"*64,
        physically_reloaded=True,frozen_parent=True,captured_before_gold=True)

class S194CertificateTests(unittest.TestCase):
    def test_full_cpu_584_plus32_hash_first_and_continuation(self):
        c=cert()
        self.assertTrue(c.compatible())
        self.assertEqual(c.old_full_eos_correct,456)
        self.assertEqual(c.new_full_eos_correct,128)
        self.assertEqual(c.train32_checked,32)
        self.assertEqual(c.train32_first_correct,16)
        self.assertEqual(c.train32_full_eos_correct,0)
        self.assertEqual(len(c.protected_584_generation_digest),64)

    def test_all_584_original_ids_must_match_posthoc_target(self):
        old,new,gold,probes=data()
        old[next(iter(old))]=[3,9,0]
        c=make_real_584_certificate(
            old_pred=old,old_target={sha(f"old{i}"):[3,4,0] for i in range(456)},
            new_pred=new,new_target=new,probe_pred=probes,probe_target=gold,
            eos_id=0,source_bpp=.69,speed_ratio=1.,
            checkpoint_sha256="a"*64,physically_reloaded=True,
            frozen_parent=True,captured_before_gold=True)
        self.assertFalse(c.compatible())
        self.assertEqual(c.old_full_lost,1)

    def test_full_generation_digest_is_order_invariant_but_token_sensitive(self):
        old,new,_,_=data()
        a=output_digest(old)
        self.assertEqual(a,output_digest(dict(reversed(list(old.items())))))
        old[next(iter(old))]=[3,8,0]
        self.assertNotEqual(a,output_digest(old))
        self.assertNotEqual(a,output_digest(new))

    def test_reject_missing_identity_duplicate_sha_unsafe_source(self):
        old,new,gold,probes=data()
        with self.assertRaises(AssertionError):
            make_real_584_certificate(
                old_pred=old,old_target=old,new_pred=new,new_target=new,
                probe_pred=probes,probe_target=dict(list(gold.items())[1:]),
                eos_id=0,source_bpp=.69,speed_ratio=1.,
                checkpoint_sha256="a"*64,physically_reloaded=True,
                frozen_parent=True,captured_before_gold=True)
        with self.assertRaises(AssertionError):
            make_real_584_certificate(
                old_pred=old,old_target=old,new_pred=new,new_target=new,
                probe_pred=probes,probe_target=gold,eos_id=0,source_bpp=.69,
                speed_ratio=1.,checkpoint_sha256="a"*64,
                physically_reloaded=True,frozen_parent=True,
                captured_before_gold=False)

    def test_actual_two_physical_fp16_safetensors_reload_and_hash(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/"a.safetensors",Path(td)/"b.safetensors"
            save_file({"up":torch.ones(2,3,dtype=torch.float16)},str(a))
            save_file({"slot":torch.zeros(4,dtype=torch.float16)},str(b))
            first=actual_fp16_checkpoint_sha([a,b])
            self.assertEqual(len(first),64)
            self.assertEqual(first,actual_fp16_checkpoint_sha([a,b]))
            save_file({"slot":torch.ones(4,dtype=torch.float16)},str(b))
            self.assertNotEqual(first,actual_fp16_checkpoint_sha([a,b]))

    def test_one_step_connector_rejects_mismatched_physical_certificate(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/"a.safetensors",Path(td)/"b.safetensors"
            p=torch.nn.Parameter(torch.tensor([.5],dtype=torch.float32))
            opt=torch.optim.Adam([p],lr=.001)
            before=p.detach().clone()
            def export():
                save_file({"up":p.detach().cpu().to(torch.float16)},str(a))
                save_file({"slot":torch.zeros(2,dtype=torch.float16)},str(b))
                return [a,b]
            def should_not_train():
                raise AssertionError("S194 must reject before any optimizer step")
            with self.assertRaisesRegex(AssertionError,"different physical weights"):
                one_step_with_actual_physical_cpu_gate(
                    params=[p],optimizer=opt,train_one_step=should_not_train,
                    physically_export_fp16=export,
                    independent_real_cpu_j7_on_export=lambda paths:cert())
            self.assertTrue(torch.equal(p.detach(),before))

if __name__=="__main__":
    unittest.main()
