"""S154 original-route isolation tests: no identity or answer based gating."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from calibri2.s145.neural_skill_blocks import LayerwiseNeuralSkillBank, NeuralSkillSpec
from calibri2.s154.route_scoped_skill_bridge import S93ScopedNeuralSkillHookBridge


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class TestS154FrozenProfileIsolation(unittest.TestCase):
    def setUp(self):
        self.model = TinyDecoder()
        self.bank = LayerwiseNeuralSkillBank(4, 2)
        self.bank.add_skill(NeuralSkillSpec("language_eos_s93_scoped_v2", (0, 1), 2))
        with torch.no_grad():
            for idx in (0, 1):
                block = self.bank.layers[str(idx)].blocks["language_eos_s93_scoped_v2"]
                block.up.weight.fill_(0.2)
        self.route = SimpleNamespace(_features=None, _active_profile=None)
        self.hooks = S93ScopedNeuralSkillHookBridge(self.bank, self.route)
        self.hooks.install(self.model)
        self.x = torch.ones(1, 3, 4)

    def tearDown(self):
        self.hooks.uninstall()

    def test_non_s93_route_is_strictly_identical_even_with_trained_skill(self):
        self.route._features = None
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.x), self.x))

    def test_original_s93_profile_uses_new_skill_only_with_prompt_features(self):
        self.route._features = torch.ones(65)
        with torch.no_grad():
            self.assertFalse(torch.equal(self.model(self.x), self.x))

    def test_s93_cached_decode_phase_preserves_route_scope(self):
        self.route._features = torch.ones(65)
        self.route._active_profile = None  # S115 clears it during cached decode
        with torch.no_grad():
            self.assertFalse(torch.equal(self.model(self.x), self.x))
        self.route._features = None
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.x), self.x))

    def test_uninstall_restores_original_decoder_even_on_s93(self):
        self.route._features = torch.ones(65)
        self.hooks.uninstall()
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.x), self.x))


if __name__ == "__main__":
    unittest.main()
