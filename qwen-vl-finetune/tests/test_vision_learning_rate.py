"""Independent vision learning rates must survive optimizer construction."""
import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Trainer


# Import the complete native trainer without requiring the CUDA FlashAttention
# extension: these optimizer tests never call its attention entry point.
flash_interface = ModuleType("flash_attn.flash_attn_interface")
flash_interface.flash_attn_varlen_func = lambda *args, **kwargs: None
previous = sys.modules.get(flash_interface.__name__)
sys.modules[flash_interface.__name__] = flash_interface
try:
    trainer = importlib.import_module("qwenvl.train.trainer")
finally:
    if previous is None:
        del sys.modules[flash_interface.__name__]
    else:
        sys.modules[flash_interface.__name__] = previous


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(2, 2)
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.visual.merger = torch.nn.Linear(2, 2)


def make_optimizer(vision_lr, projector_lr, frozen=False):
    model = TinyModel()
    if frozen:
        model.visual.blocks[0].bias.requires_grad_(False)
    context = SimpleNamespace(
        model=model, optimizer=None,
        args=SimpleNamespace(
            vision_tower_lr=vision_lr, mm_projector_lr=projector_lr,
            weight_decay=0.2,
        ),
        get_decay_parameter_names=lambda model: [
            name for name, parameter in model.named_parameters() if parameter.ndim == 2
        ],
    )
    with patch.object(Trainer, "get_optimizer_cls_and_kwargs",
                      return_value=(torch.optim.SGD, {"lr": 0.01})):
        optimizer = trainer.create_optimizer(context)
    return model, optimizer


class VisionLearningRateTest(unittest.TestCase):
    def assert_rates(self, vision_lr, projector_lr, frozen=False):
        model, optimizer = make_optimizer(vision_lr, projector_lr, frozen)
        by_id = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertNotIn(id(parameter), by_id)
                by_id[id(parameter)] = group
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertNotIn(id(parameter), by_id)
                continue
            group = by_id[id(parameter)]
            expected_lr = 0.01
            if "merger" in name and projector_lr:
                expected_lr = projector_lr
            elif "visual" in name and "merger" not in name and vision_lr:
                expected_lr = vision_lr
            self.assertEqual(group["lr"], expected_lr, name)
            self.assertEqual(group["weight_decay"], 0.2 if parameter.ndim == 2 else 0.0, name)
            parameter.data.fill_(1)
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                group = by_id[id(parameter)]
                expected = 1 - group["lr"] * (1 + group["weight_decay"])
                torch.testing.assert_close(parameter, torch.full_like(parameter, expected))

    def test_vision_override_without_projector_override(self):
        for projector_lr in (None, 0.0):
            with self.subTest(projector_lr=projector_lr):
                self.assert_rates(0.002, projector_lr)

    def test_separate_vision_and_projector_overrides(self):
        self.assert_rates(0.002, 0.003)

    def test_projector_only_and_default_rates(self):
        self.assert_rates(None, 0.003)
        self.assert_rates(None, None)
        self.assert_rates(0.0, 0.0)

    def test_frozen_parameters_are_excluded(self):
        self.assert_rates(0.002, None, frozen=True)


if __name__ == "__main__":
    unittest.main()

