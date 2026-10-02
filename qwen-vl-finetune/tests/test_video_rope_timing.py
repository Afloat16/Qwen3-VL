"""Use processor timing for every Qwen2.5-VL video in a training sample."""
from types import SimpleNamespace
import unittest

import torch

from qwenvl.data.data_processor import LazySupervisedDataset
from qwenvl.data.rope2d import get_rope_index_25


class RecordingProcessor:
    def __init__(self, timing, two_videos=True):
        self.timing = timing
        self.two_videos = two_videos
        self.video_processor = SimpleNamespace(temporal_patch_size=2, fps=2.0)
        self.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *args, **kwargs: "")

    def apply_chat_template(self, messages, **kwargs):
        ids = [10, 151652, 151656, 151656, 151656, 151653, 11]
        grids = [[3, 2, 2]]
        if self.two_videos:
            ids += [151652, 151656, 151656, 151656, 151653, 12]
            grids += [[3, 2, 2]]
        result = {"input_ids": torch.tensor([ids]), "video_grid_thw": torch.tensor(grids)}
        if self.timing is not None:
            result["second_per_grid_ts"] = self.timing
        return result


def sample(timing, two_videos=True):
    processor = RecordingProcessor(timing, two_videos)
    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.processor = processor
    dataset.merge_size = 2
    dataset.get_rope_index = get_rope_index_25
    source = {
        "video": ["first.mp4", "second.mp4"] if two_videos else ["first.mp4"],
        "conversations": [{"from": "human", "value": "<video><video>" if two_videos else "<video>"}],
    }
    return dataset._get_item([source])


class VideoTimingTest(unittest.TestCase):
    def test_independent_timing_for_two_videos(self):
        result = sample([4.0, 0.5])
        expected = torch.tensor([
            [0, 1, 2, 10, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [0, 1, 2, 2, 2, 19, 20, 21, 22, 22, 22, 25, 26],
            [0, 1, 2, 2, 2, 19, 20, 21, 22, 22, 22, 25, 26],
        ]).unsqueeze(1)
        torch.testing.assert_close(result["position_ids"], expected)
        self.assertEqual(result["attention_mask"], [13])

    def test_single_video_preserves_tensor_timing(self):
        result = sample(torch.tensor([2.0]), two_videos=False)
        expected = torch.tensor([
            [0, 1, 2, 6, 10, 11, 12],
            [0, 1, 2, 2, 2, 11, 12],
            [0, 1, 2, 2, 2, 11, 12],
        ]).unsqueeze(1)
        torch.testing.assert_close(result["position_ids"], expected)

    def test_fallback_counts_videos_not_container_tensors(self):
        result = sample(None)
        expected = torch.tensor([
            [0, 1, 2, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16],
            [0, 1, 2, 2, 2, 7, 8, 9, 10, 10, 10, 15, 16],
            [0, 1, 2, 2, 2, 7, 8, 9, 10, 10, 10, 15, 16],
        ]).unsqueeze(1)
        torch.testing.assert_close(result["position_ids"], expected)


if __name__ == "__main__":
    unittest.main()

