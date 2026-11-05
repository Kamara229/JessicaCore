import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from direct_codebook_v59 import encode, CB59
from direct_runtime_v59 import NativeKernel, compile_kernel
from v173.recover import exact_dense_matrix
from v176.operators import TrainWeight, TrainLinear, TrainEmbedding, fp16_ste
from v176.codec import replace_fields
from v176.objective import hard_margin
from v176.data import instruction_pairs
from v169.sources import questions


def source():
    torch.manual_seed(176)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.head = nn.Linear(16, 32, bias=False); self.head.weight = self.embed.weight
            self.norm = nn.LayerNorm(16)
    return CB59(encode(Toy().state_dict())[0])


class JointTraining(unittest.TestCase):
    def test_literal_surrogate_and_gradients_both_tied_uses(self):
        archive = source(); handle = archive.handles['embed.weight']
        w = TrainWeight(handle); emb = TrainEmbedding(w); head = TrainLinear(w)
        np.testing.assert_array_equal(w().detach().numpy(), exact_dense_matrix(handle))
        head(emb(torch.tensor([1, 3, 5]))).square().mean().backward()
        self.assertIs(emb.source, head.source)
        self.assertGreater(float(w.centroids.grad.abs().sum()), 0)
        self.assertGreater(float(w.log_scales.grad.abs().sum()), 0)
        old = w().detach().clone()
        with torch.no_grad(): w.centroids.add_(0.07)
        w.clear()
        self.assertFalse(torch.equal(old, w()))

    def test_fp16_is_executed_with_straight_through_gradient(self):
        x = torch.tensor([0.123456, -2.123456], requires_grad=True)
        y = fp16_ste(x)
        self.assertTrue(torch.equal(y, x.half().float()))
        y.sum().backward(); self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_hard_margin_excludes_target_and_masks_prompt(self):
        logits = torch.tensor([[[4., 2., 1.], [1., 5., 3.], [9., 1., 0.]]], requires_grad=True)
        target = torch.tensor([[0, 2, 2]])
        mask = torch.tensor([[True, True, False]])
        loss = hard_margin(logits, target, mask)
        self.assertAlmostEqual(float(loss), 1.5)
        loss.backward(); self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)

    def test_changed_physical_fields_reload_native_and_alias(self):
        archive = source(); values = {}
        for name, d in archive.meta['tensor_descriptors'].items():
            if d['dtype'] == 'CB59_K32V8':
                h = archive.handles[name]
                values[name] = {'centroids': h.centroids + .1, 'row_scales': h.row_scales * 1.03}
            else: values[name] = {'raw': archive.raw[name] + .05}
        saved = CB59(replace_fields(archive, values))
        self.assertEqual(len(saved.blob), len(archive.blob))
        self.assertIs(saved.handles['embed.weight'], saved.handles['head.weight'])
        h = saved.handles['embed.weight']
        with TemporaryDirectory() as tmp:
            lib = NativeKernel(compile_kernel(Path(__file__).with_name('direct_kernel_v59.c'), Path(tmp)/'kernel.so'))
            x = np.random.default_rng(176).normal(size=(3, 16)).astype(np.float32)
            np.testing.assert_allclose(lib.linear(h, x), x @ exact_dense_matrix(h).T, atol=2e-5, rtol=1e-5)
        with self.assertRaises(ValueError): replace_fields(archive, {})
        first = next(n for n, d in archive.meta['tensor_descriptors'].items() if d['dtype'] == 'CB59_K32V8')
        values[first]['centroids'][0, 0] = np.nan
        with self.assertRaises(ValueError): replace_fields(archive, values)

    def test_training_questions_not_historical_evaluation(self):
        train = instruction_pairs(); historical = questions()
        self.assertFalse({q for q, _ in train} & {q.question for q in historical})
        self.assertEqual(len(train), 32)
        self.assertEqual(sum(a == 'Yes' for _, a in train), 4)
        self.assertEqual(sum(a == 'No' for _, a in train), 4)


if __name__ == '__main__': unittest.main()
