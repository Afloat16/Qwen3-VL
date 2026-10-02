"""CPU checks for independent fine-tuning component learning rates.

Only the exact optimizer function is loaded from the training helper to avoid
its unrelated CUDA-only FlashAttention import. Models and Trainer are native.
"""

import ast
import itertools
import os
from pathlib import Path

import pytest
import torch
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(
    os.environ.get("QWEN_OPTIMIZER_SOURCE_ROOT", Path(__file__).resolve().parents[2])
)


def load_optimizer(root):
    path = root / "qwen-vl-finetune/qwenvl/train/trainer.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_optimizer"
    )
    namespace = {"Trainer": Trainer}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["create_optimizer"]


create_optimizer = load_optimizer(ROOT)


def make_model():
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config)


def make_trainer(model, tmp_path, projector_lr, vision_lr, weight_decay=0.1):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        use_cpu=True,
        optim="sgd",
        learning_rate=0.05,
        weight_decay=weight_decay,
    )
    args.mm_projector_lr = projector_lr
    args.vision_tower_lr = vision_lr
    return Trainer(model=model, args=args)


def parameter_groups(optimizer):
    return {id(p): group for group in optimizer.param_groups for p in group["params"]}


def rate_or_default(value):
    return 0.05 if value is None or value == 0 else value


@pytest.mark.parametrize("projector_lr", [None, 0.0, 0.02])
@pytest.mark.parametrize("vision_lr", [None, 0.0, 0.005])
def test_independent_component_rates_and_weight_decay(
    projector_lr, vision_lr, tmp_path
):
    model = make_model()
    trainer = make_trainer(model, tmp_path, projector_lr, vision_lr)
    optimizer = create_optimizer(trainer)
    groups = parameter_groups(optimizer)
    decay = {
        name for name in trainer.get_decay_parameter_names(model) if "bias" not in name
    }
    assert len(groups) == sum(len(group["params"]) for group in optimizer.param_groups)
    for name, parameter in model.named_parameters():
        if "merger" in name:
            expected_lr = rate_or_default(projector_lr)
        elif "visual" in name:
            expected_lr = rate_or_default(vision_lr)
        else:
            expected_lr = 0.05
        assert groups[id(parameter)]["lr"] == expected_lr, name
        assert groups[id(parameter)]["weight_decay"] == (
            0.1 if name in decay else 0.0
        ), name


@pytest.mark.parametrize("projector_lr", [None, 0.0, 0.02])
def test_requested_encoder_rate_controls_an_actual_sgd_step(projector_lr, tmp_path):
    model = make_model()
    trainer = make_trainer(model, tmp_path, projector_lr, 0.005, weight_decay=0.0)
    optimizer = create_optimizer(trainer)
    parameter = model.visual.blocks[0].attn.qkv.weight
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, before - 0.005)


@pytest.mark.parametrize(
    "vision,projector,llm", list(itertools.product([False, True], repeat=3))
)
@pytest.mark.parametrize(
    "projector_lr,vision_lr", [(None, 0.005), (0.02, None), (0.02, 0.005)]
)
def test_groups_partition_trainable_parameters(
    vision, projector, llm, projector_lr, vision_lr, tmp_path
):
    model = make_model()
    for name, parameter in model.named_parameters():
        if "merger" in name:
            parameter.requires_grad_(projector)
        elif "visual" in name:
            parameter.requires_grad_(vision)
        else:
            parameter.requires_grad_(llm)
    optimizer = create_optimizer(make_trainer(model, tmp_path, projector_lr, vision_lr))
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


def test_existing_optimizer_is_returned_unchanged(tmp_path):
    model = make_model()
    trainer = make_trainer(model, tmp_path, None, 0.005)
    first = create_optimizer(trainer)
    trainer.args.vision_tower_lr = 0.03
    assert create_optimizer(trainer) is first
