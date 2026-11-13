"""S204 frozen S73 source-context contracts; mocked hooks, NOT real CPU J7."""
from contextlib import contextmanager
import sys
import types
import unittest
from unittest.mock import patch

from calibri2.s204.test_protected_s73_train import S204OriginalS73SourceTests
from calibri2.s204.original_train_source_scope import (
    ensure_original_s73_source,
    make_original_s73_training_scope,
)


class S204FrozenS73ProfileTests(unittest.TestCase):
    def setup_train(self):
        old = S204OriginalS73SourceTests.old_original()
        routes = {k: "S93" for k in old}
        return old, routes

    def test_precommitted_sha_and_original_route_required(self):
        old, routes = self.setup_train()
        identity = "S73/73012/left"
        ensure_original_s73_source(
            identity=identity, route="S93", row=old[identity],
            source_routes=routes)
        for route in ("S87", "INVALID"):
            with self.assertRaises(AssertionError):
                ensure_original_s73_source(
                    identity=identity, route=route, row=old[identity],
                    source_routes=routes)
        with self.assertRaises(AssertionError):
            ensure_original_s73_source(
                identity=identity, route="S93",
                row={"question":old[identity]["question"]+" "},
                source_routes=routes)

    def test_missing_one_train_route_fails_closed(self):
        old, routes = self.setup_train()
        del routes["S73/73015/right"]
        with self.assertRaises(AssertionError):
            make_original_s73_training_scope(
                old_456=old,source_routes=routes,cpu=(),tok=None,
                bank={},bridge=None,full=None,expansion=None)

    def test_original_real_prefill_hooks_and_profile_restore(self):
        old, routes = self.setup_train()
        identity = "S73/73012/left"
        events = []
        s160 = types.ModuleType("calibri2.s160.real_parallel_slots_j7")
        s160.prompt_only_vector = lambda tok, question: [0.1, 0.2]
        s160.apply_profile = lambda *_args: events.append(
            ("route",_args[2]))
        q2 = types.ModuleType("calibri2.s127.q2_prefill_gate_control")
        @contextmanager
        def prefill_full(model, full):
            events.append(("GT", "enter"))
            try: yield
            finally: events.append(("GT","exit"))
        q2.prefill_only_tunnel = prefill_full
        ops = types.ModuleType("v176.operators")
        ops.clear = lambda sources: events.append(("cache","clear"))
        @contextmanager
        def expansion_prefill(model):
            events.append(("growth","enter"))
            try: yield
            finally: events.append(("growth","exit"))
        @contextmanager
        def bridge_prompt(route, features):
            events.append(("bridge",route))
            self.assertEqual(tuple(features.tolist()),(0.10000000149011612,0.20000000298023224))
            try: yield
            finally: events.append(("bridge","exit"))
        model,sources = object(),object()
        cpu=(model,sources,None,None,None,None,object(),object(),object())
        b = types.SimpleNamespace(_handles=[1],prompt_scope=bridge_prompt)
        f = types.SimpleNamespace(_handles=[1])
        e = types.SimpleNamespace(_handles=[1],prefill_only=expansion_prefill)
        fake_modules = {
            "calibri2.s160.real_parallel_slots_j7":s160,
            "calibri2.s127.q2_prefill_gate_control":q2,
            "v176.operators":ops,
        }
        with patch.dict(sys.modules,fake_modules):
            mk = make_original_s73_training_scope(
                old_456=old,source_routes=routes,cpu=cpu,tok=object(),
                bank={},bridge=b,full=f,expansion=e)
            with mk(identity,"S93"):
                events.append(("TRAIN","gradient"))
        self.assertEqual(events[0],("route","S93"))
        self.assertIn(("TRAIN","gradient"),events)
        self.assertLess(events.index(("GT","enter")),events.index(("TRAIN","gradient")))
        self.assertLess(events.index(("bridge","S93")),events.index(("TRAIN","gradient")))
        self.assertEqual(events[-2:], [("route","S93"),("cache","clear")])

    def test_inference_mode_cannot_compute_s73_gradient(self):
        old,routes=self.setup_train()
        mk = make_original_s73_training_scope(
            old_456=old,source_routes=routes,
            cpu=(None,)*9,tok=None,bank={},
            bridge=types.SimpleNamespace(_handles=[1]),
            full=types.SimpleNamespace(_handles=[1]),
            expansion=types.SimpleNamespace(_handles=[1]))
        import torch
        with torch.inference_mode():
            with self.assertRaises(AssertionError):
                with mk("S73/73012/left","S93"): pass


if __name__ == "__main__":
    unittest.main()
