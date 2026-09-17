import importlib.util
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "operators" / "hand_pose" / "estimate_hands.py"
)
SPEC = importlib.util.spec_from_file_location("estimate_hands", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HandPoseOperatorTests(unittest.TestCase):
    def test_interpolates_short_bbox_gap(self):
        frames = [
            {
                "frame_index": index,
                "hands": (
                    {
                        "left": {
                            "bbox_xyxy": [index, index, index + 10, index + 10],
                            "score": 0.9,
                        }
                    }
                    if index in (0, 2)
                    else {}
                ),
            }
            for index in range(3)
        ]

        segments = MODULE.build_track_segments(frames, "left", max_gap=2)

        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0].frame_indices, [0, 1, 2])
        np.testing.assert_allclose(segments[0].boxes[1], [1, 1, 11, 11])
        np.testing.assert_array_equal(segments[0].detected, [True, False, True])

    def test_does_not_bridge_long_gap(self):
        frames = [
            {
                "frame_index": index,
                "hands": (
                    {"right": {"bbox_xyxy": [0, 0, 10, 10], "score": 0.8}}
                    if index in (0, 10)
                    else {}
                ),
            }
            for index in range(11)
        ]

        segments = MODULE.build_track_segments(frames, "right", max_gap=3)

        self.assertEqual(len(segments), 2)
        self.assertEqual([len(segment.frame_indices) for segment in segments], [1, 1])

    def test_sequence_chunk_is_padded_to_hawor_length(self):
        segment = MODULE.TrackSegment(
            hand="left",
            frame_indices=np.arange(3),
            boxes=np.tile([1, 2, 10, 20], (3, 1)).astype(np.float32),
            detected=np.ones(3, dtype=bool),
            scores=np.ones(3, dtype=np.float32),
        )

        chunks = MODULE.make_sequence_chunks([segment])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].valid_count, 3)
        self.assertEqual(len(chunks[0].frame_indices), MODULE.SEQUENCE_LENGTH)
        self.assertEqual(chunks[0].frame_indices[-1], 2)

    def test_renderer_draws_box_and_joints(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        points = [[30 + index * 2, 50 + index] for index in range(21)]
        hands = [
            {
                "handedness": "left",
                "bbox_xyxy": [20, 20, 100, 110],
                "detection_score": 0.9,
                "keypoints_2d": points,
            }
        ]

        rendered = MODULE.render_hand_pose(frame, hands)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(cv2.countNonZero(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)), 0)


if __name__ == "__main__":
    unittest.main()
