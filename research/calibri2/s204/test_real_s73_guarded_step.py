"""Unit-only S204 source-route wiring; NOT a native 616 CPU J7 execution."""
from contextlib import contextmanager
from unittest.mock import patch
import unittest
import torch

from calibri2.s204.real_s73_guarded_step import real_four_s73_guarded_step
from calibri2.s204.test_protected_s73_train import S204OriginalS73SourceTests


class TinyArea(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.decoder.weight.fill_(1.0)
        self._handles = [object()]
        self._active = True


class S204RealTRAINWiring(unittest.TestCase):
    def make(self):
        area = TinyArea()
        optimizer = torch.optim.AdamW(area.parameters(), lr=0.05, weight_decay=0)
        model = torch.nn.Linear(1, 1).requires_grad_(False)
        old = S204OriginalS73SourceTests.old_original()
        routes = {k: "S93" for k in old}
        return area, optimizer, model, old, routes

    def test_exactly_four_original_s73_gradients_are_passed_to_adam(self):
        area, opt, model, old, routes = self.make()
        touched = []
        state = {"inside": None}
        @contextmanager
        def original_profile(identity, route):
            self.assertEqual(route, "S93")
            self.assertIsNone(state["inside"])
            state["inside"] = identity
            touched.append(identity)
            try:
                yield
            finally:
                state["inside"] = None

        def synthetic_unit_gradient(**kwargs):
            self.assertIsNotNone(state["inside"])
            self.assertEqual(kwargs["first_token_id"], 23429)
            self.assertEqual(kwargs["question"], old[state["inside"]]["question"])
            # Unit-only contract stub. Real training is required to call
            # the genuine real_s73_nra_gradient; never call this a CPU J7.
            return {"neuro_representation_area.decoder.weight":
                    torch.ones_like(area.decoder.weight)}

        with patch("calibri2.s204.real_s73_guarded_step.real_s73_nra_gradient",
                   side_effect=synthetic_unit_gradient) as mocked:
            proof = real_four_s73_guarded_step(
                original_456=old, source_routes=routes,
                source_scope=original_profile, model=model, cpu=(None,)*6,
                tok=object(), area=area, optimizer=opt,
                backward_new32_train_loss=lambda: (area.decoder.weight ** 2).mean())
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(set(touched), {
            "S73/73012/left","S73/73012/right",
            "S73/73015/left","S73/73015/right"})
        self.assertEqual(proof["protected_first_token_train_examples"], 4)
        self.assertTrue(proof["original_456_new128_novel32_real_CPU_J7_pending"])
        self.assertFalse(proof["model_accepted"])
        self.assertLess(float(area.decoder.weight), 1.0)
        self.assertIsNone(state["inside"])

    def test_missing_frozen_route_fails_before_optimizer(self):
        area, opt, model, old, routes = self.make()
        del routes["S73/73012/left"]
        with self.assertRaises(AssertionError):
            real_four_s73_guarded_step(
                original_456=old, source_routes=routes,
                source_scope=lambda *a: None, model=model, cpu=(None,)*6,
                tok=object(), area=area, optimizer=opt,
                backward_new32_train_loss=lambda: area.decoder.weight.sum())
        self.assertFalse(opt.state)

    def test_unfrozen_core_fails(self):
        area, opt, model, old, routes = self.make()
        model.requires_grad_(True)
        with self.assertRaises(AssertionError):
            real_four_s73_guarded_step(
                original_456=old, source_routes=routes,
                source_scope=lambda *a: None, model=model, cpu=(None,)*6,
                tok=object(), area=area, optimizer=opt,
                backward_new32_train_loss=lambda: area.decoder.weight.sum())
        self.assertFalse(opt.state)


if __name__ == "__main__":
    unittest.main()
