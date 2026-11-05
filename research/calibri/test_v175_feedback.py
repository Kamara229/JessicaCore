import unittest
from types import SimpleNamespace
import numpy as np
from v175.feedback import metric_and_factor, conditional_codes, fit_indices
from v175.protocol import calibration_key

class FeedbackMath(unittest.TestCase):
    def test_block_update_is_constrained_quadratic_optimum(self):
        rng=np.random.default_rng(175)
        a=rng.normal(size=(24,24));h=a.T@a+np.eye(24)*0.1
        inv=np.linalg.inv(h);u=np.linalg.cholesky(inv).T
        error=rng.normal(size=(5,8))
        update=-error@np.linalg.inv(u[:8,:8])@u[:8,8:]
        direct=error@h[:8,8:]@np.linalg.inv(h[8:,8:])
        np.testing.assert_allclose(update,direct,atol=1e-10)
        residual=np.concatenate((error,-update),axis=1)
        energy=np.einsum('bi,ij,bj->b',residual,h,residual)
        np.testing.assert_allclose(energy,np.sum((error@np.linalg.inv(u[:8,:8]))**2,axis=1),atol=1e-10)
        # Sign matters: weight update is the negative of remaining residual.
        wrong=np.concatenate((error,update),axis=1)
        self.assertTrue(np.all(np.einsum('bi,ij,bj->b',wrong,h,wrong)>energy))

    def test_diagonal_negative_control_matches_independent_assignment(self):
        rng=np.random.default_rng(176)
        w=rng.normal(size=(32,24));c=rng.normal(size=(32,8))
        h,u=metric_and_factor(np.diag(np.arange(1,25)))
        codes=conditional_codes(w,c,u)
        for g in range(3):
            d=np.diag(h)[g*8:g*8+8]
            expected=np.argmin(np.sum((w[:,None,g*8:g*8+8]-c[None])**2*d,axis=2),axis=1)
            np.testing.assert_array_equal(codes[:,g],expected)

    def test_exact_codebook_source_preserved_and_fallback_nonincreasing(self):
        rng=np.random.default_rng(177)
        c=rng.normal(size=(32,8)).astype(np.float16).astype(np.float32)
        ids=rng.integers(0,32,size=(32,3),dtype=np.uint8)
        w=c[ids].reshape(32,24)
        a=rng.normal(size=(40,24));h=a.T@a
        handle=SimpleNamespace(shape=w.shape,row_scales=np.ones(32,dtype=np.float32),centroids=c)
        selected,report=fit_indices(w,handle,ids,h)
        np.testing.assert_array_equal(selected,ids)
        noisy=w+0.1*rng.normal(size=w.shape)
        _,r=fit_indices(noisy,handle,ids,h,tied=True)
        self.assertLessEqual(r['fit_quadratic_selected'],r['fit_quadratic_before'])

    def test_invalid_covariance_and_alias_mapping(self):
        with self.assertRaises(ValueError):metric_and_factor(np.zeros((8,8)))
        with self.assertRaises(ValueError):metric_and_factor(np.ones((7,7)))
        h=np.eye(8);h[0,1]=1
        with self.assertRaises(ValueError):metric_and_factor(h)
        self.assertEqual(calibration_key('model.layers.4.self_attn.k_proj.weight'),'model.layers.4.self_attn.q_proj')
        self.assertEqual(calibration_key('model.embed_tokens.weight'),'lm_head')

class ArchiveNative(unittest.TestCase):
    def test_changed_codes_preserve_bytes_and_native_numeric_parity(self):
        import torch
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from direct_codebook_v59 import encode,CB59,unpack5
        from direct_runtime_v59 import NativeKernel,compile_kernel
        from v173.recover import exact_dense_matrix
        from v175.codec import replace_indices
        torch.manual_seed(175)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed=torch.nn.Embedding(32,16)
                self.head=torch.nn.Linear(16,32,bias=False);self.head.weight=self.embed.weight
        toy=Toy();blob,_,_=encode(toy.state_dict());old=CB59(blob)
        assignments={}
        for n,d in old.meta['tensor_descriptors'].items():
            if d['dtype']=='CB59_K32V8':
                h=old.handles[n]
                ids=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//8).reshape(h.shape[0],-1)
                assignments[n]=(ids+1)%32
        changed=replace_indices(old,assignments);new=CB59(changed)
        self.assertEqual(len(blob),len(changed));self.assertNotEqual(blob,changed)
        self.assertIs(new.handles['embed.weight'],new.handles['head.weight'])
        h=new.handles['embed.weight']
        np.testing.assert_array_equal(h.centroids,old.handles['embed.weight'].centroids)
        np.testing.assert_array_equal(h.row_scales,old.handles['embed.weight'].row_scales)
        with TemporaryDirectory() as tmp:
            lib=NativeKernel(compile_kernel(Path(__file__).with_name('direct_kernel_v59.c'),Path(tmp)/'kernel.so'))
            x=np.random.default_rng(175).normal(size=(3,16)).astype(np.float32)
            np.testing.assert_allclose(lib.linear(h,x),x@exact_dense_matrix(h).T,atol=2e-5,rtol=1e-5)
        with self.assertRaises(ValueError):replace_indices(old,{})
        assignments[next(iter(assignments))][0,0]=32
        with self.assertRaises(ValueError):replace_indices(old,assignments)

if __name__=='__main__':unittest.main()
