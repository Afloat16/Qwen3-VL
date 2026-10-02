"""COCO evaluation must retain metrics when the model detects no objects."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

spec = importlib.util.spec_from_file_location(
    "odinw_eval_utils",
    Path(__file__).resolve().parents[1] / "evaluation/ODinW-13/eval_utils.py",
)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)


def make_coco(with_ground_truth=True):
    coco = COCO()
    coco.dataset = {
        "info": {},
        "images": [{"id": 7, "height": 100, "width": 100}],
        "categories": [{"id": 11, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 11,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ]
        if with_ground_truth
        else [],
    }
    coco.createIndex()
    return coco


def empty_prediction():
    return (
        {"img_id": 7},
        {
            "img_id": 7,
            "labels": np.empty(0, dtype=int),
            "bboxes": np.empty((0, 4)),
            "scores": np.empty(0),
        },
    )


@pytest.mark.parametrize("results", [[], [empty_prediction()]])
def test_empty_detections_receive_zero_average_precision(results):
    metrics = eval_utils.compute_metrics(results, _coco_api=make_coco())
    assert metrics == {
        "mAP": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_s": 0.0,
        "mAP_m": -1.0,
        "mAP_l": -1.0,
    }


def test_no_ground_truth_preserves_coco_undefined_values():
    metrics = eval_utils.compute_metrics([], _coco_api=make_coco(False))
    assert metrics == {
        name: -1.0 for name in ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
    }


def test_nonempty_perfect_detection_remains_perfect():
    pred = {
        "img_id": 7,
        "labels": np.array([0]),
        "bboxes": np.array([[10, 10, 30, 30]]),
        "scores": np.array([1.0]),
    }
    metrics = eval_utils.compute_metrics([({"img_id": 7}, pred)], _coco_api=make_coco())
    assert metrics["mAP"] == metrics["mAP_50"] == metrics["mAP_75"] == 1.0
