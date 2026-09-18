#!/usr/bin/env python3
"""Estimate 21 MANO hand joints with HaWoR and render them onto a video."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SEQUENCE_LENGTH = 16
DEFAULT_HAWOR_ROOT = (
    Path.home() / "Desktop" / "DemoAI-TrainingData" / "models" / "hawor" / "runtime"
)
HAND_NAMES = {0: "left", 1: "right"}
HAND_COLORS = {
    "left": (255, 180, 30),
    "right": (50, 210, 90),
}
FINGER_COLORS = [
    (80, 200, 255),
    (80, 255, 120),
    (255, 220, 70),
    (255, 130, 80),
    (220, 90, 255),
]
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
JOINT_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class TrackSegment:
    hand: str
    frame_indices: np.ndarray
    boxes: np.ndarray
    detected: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class SequenceChunk:
    hand: str
    frame_indices: np.ndarray
    boxes: np.ndarray
    valid_count: int


def input_video(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input video does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect hands, run HaWoR temporal pose estimation, decode 21 MANO "
            "joints, and render the result to a new video."
        )
    )
    parser.add_argument("input", type=input_video, help="local input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: INPUT_PARENT/INPUT_STEM_hand_pose)",
    )
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=DEFAULT_HAWOR_ROOT,
        help=f"official HaWoR checkout (default: {DEFAULT_HAWOR_ROOT})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--det-confidence",
        type=float,
        default=0.2,
        help="YOLO hand detection threshold (default: 0.2)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=30,
        help="maximum missed-detection gap to interpolate, in frames (default: 30)",
    )
    parser.add_argument(
        "--chunk-batch",
        type=int,
        default=4,
        help="number of 16-frame HaWoR sequences per forward pass (default: 4)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        help="camera focal length in pixels (default: image diagonal)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="process only the first N frames; intended for validation runs",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep extracted JPEG frames after successful completion",
    )
    parser.add_argument(
        "--force-detect",
        action="store_true",
        help="ignore a compatible cached detection pass",
    )
    return parser


def select_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_hawor_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    required = [
        resolved / "weights" / "external" / "detector.pt",
        resolved / "weights" / "hawor" / "model_config.yaml",
        resolved / "weights" / "hawor" / "checkpoints" / "hawor.ckpt",
        resolved / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
        resolved / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "HaWoR runtime is incomplete; missing:\n  " + "\n  ".join(missing)
        )
    return resolved


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def frame_path(frame_dir: Path, frame_index: int) -> Path:
    return frame_dir / f"{frame_index:08d}.jpg"


def atomic_json_write(document: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def read_video_info(video_path: Path, max_frames: Optional[int] = None) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def compatible_detection_cache(
    cache_path: Path,
    frame_dir: Path,
    video_path: Path,
    info: VideoInfo,
    confidence: float,
) -> Optional[Dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = cached.get("metadata", {})
    matches = (
        metadata.get("source") == str(video_path)
        and metadata.get("frame_count") == info.frame_count
        and math.isclose(float(metadata.get("det_confidence", -1)), confidence)
        and len(cached.get("frames", [])) == info.frame_count
        and frame_path(frame_dir, 0).is_file()
        and frame_path(frame_dir, info.frame_count - 1).is_file()
    )
    return cached if matches else None


def detect_and_extract(
    video_path: Path,
    info: VideoInfo,
    frame_dir: Path,
    cache_path: Path,
    detector_path: Path,
    confidence: float,
    device: str,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    detector = YOLO(str(detector_path))
    frames: List[Dict[str, Any]] = []

    print(f"Detecting hands in {info.frame_count} frames on {device} ...")
    for frame_index in range(info.frame_count):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"video ended unexpectedly at frame {frame_index}")
        if not cv2.imwrite(
            str(frame_path(frame_dir, frame_index)),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            capture.release()
            raise RuntimeError(f"failed to cache frame {frame_index}")

        result = detector.track(
            frame,
            conf=confidence,
            persist=True,
            verbose=False,
            device=device,
        )[0]
        hands: Dict[str, Dict[str, Any]] = {}
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else np.full(len(boxes), -1, dtype=int)
            )
            for class_id in (0, 1):
                candidates = np.where(classes == class_id)[0]
                if not len(candidates):
                    continue
                chosen = int(candidates[np.argmax(scores[candidates])])
                hands[HAND_NAMES[class_id]] = {
                    "bbox_xyxy": boxes[chosen].round(4).tolist(),
                    "score": round(float(scores[chosen]), 6),
                    "track_id": int(track_ids[chosen]),
                }
        frames.append({"frame_index": frame_index, "hands": hands})

        if (frame_index + 1) % 250 == 0 or frame_index + 1 == info.frame_count:
            print(f"  detection: {frame_index + 1}/{info.frame_count}")

    capture.release()
    document = {
        "metadata": {
            "source": str(video_path),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "det_confidence": confidence,
        },
        "frames": frames,
    }
    atomic_json_write(document, cache_path)
    return document


def build_track_segments(
    detection_frames: Sequence[Dict[str, Any]],
    hand: str,
    max_gap: int,
) -> List[TrackSegment]:
    detected_indices: List[int] = []
    detected_boxes: List[List[float]] = []
    detected_scores: List[float] = []
    for frame in detection_frames:
        observation = frame.get("hands", {}).get(hand)
        if observation is None:
            continue
        detected_indices.append(int(frame["frame_index"]))
        detected_boxes.append(observation["bbox_xyxy"])
        detected_scores.append(float(observation["score"]))

    if not detected_indices:
        return []

    indices = np.asarray(detected_indices, dtype=np.int64)
    boxes = np.asarray(detected_boxes, dtype=np.float32)
    scores = np.asarray(detected_scores, dtype=np.float32)
    split_points = np.where(np.diff(indices) > max_gap + 1)[0] + 1
    index_groups = np.split(np.arange(len(indices)), split_points)
    segments: List[TrackSegment] = []

    for group in index_groups:
        observed_frames = indices[group]
        observed_boxes = boxes[group]
        observed_scores = scores[group]
        full_frames = np.arange(observed_frames[0], observed_frames[-1] + 1)
        full_boxes = np.column_stack(
            [
                np.interp(full_frames, observed_frames, observed_boxes[:, coordinate])
                for coordinate in range(4)
            ]
        ).astype(np.float32)
        detected = np.isin(full_frames, observed_frames)
        full_scores = np.full(len(full_frames), np.nan, dtype=np.float32)
        positions = np.searchsorted(full_frames, observed_frames)
        full_scores[positions] = observed_scores
        segments.append(
            TrackSegment(
                hand=hand,
                frame_indices=full_frames,
                boxes=full_boxes,
                detected=detected,
                scores=full_scores,
            )
        )
    return segments


def make_sequence_chunks(segments: Sequence[TrackSegment]) -> List[SequenceChunk]:
    chunks: List[SequenceChunk] = []
    for segment in segments:
        for start in range(0, len(segment.frame_indices), SEQUENCE_LENGTH):
            frames = segment.frame_indices[start : start + SEQUENCE_LENGTH]
            boxes = segment.boxes[start : start + SEQUENCE_LENGTH]
            valid_count = len(frames)
            if valid_count < SEQUENCE_LENGTH:
                pad_count = SEQUENCE_LENGTH - valid_count
                frames = np.pad(frames, (0, pad_count), mode="edge")
                boxes = np.pad(boxes, ((0, pad_count), (0, 0)), mode="edge")
            chunks.append(
                SequenceChunk(
                    hand=segment.hand,
                    frame_indices=frames,
                    boxes=boxes,
                    valid_count=valid_count,
                )
            )
    return chunks


def load_hawor_model(hawor_root: Path, device: str):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(hawor_root))

    import torch
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR

    config_path = hawor_root / "weights" / "hawor" / "model_config.yaml"
    checkpoint_path = hawor_root / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
    config = get_config(str(config_path), update_cachedir=True)
    if config.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in config.MODEL:
        config.defrost()
        config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()

    print(f"Loading HaWoR checkpoint on {device} ...")
    model = HAWOR.load_from_checkpoint(
        str(checkpoint_path),
        strict=False,
        cfg=config,
        map_location="cpu",
    )
    model = model.to(torch.device(device)).eval()
    return model


def infer_hand(
    model,
    chunks: Sequence[SequenceChunk],
    frame_dir: Path,
    focal_length: float,
    image_center: Tuple[float, float],
    device: str,
    chunk_batch: int,
) -> Dict[int, Dict[str, Any]]:
    import torch
    from torch.utils.data import default_collate
    from hawor.utils.rotation import (
        angle_axis_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
    )
    from lib.datasets.track_dataset import TrackDatasetEval
    from lib.models.mano_wrapper import MANO

    predictions: Dict[int, Dict[str, Any]] = {}
    if not chunks:
        return predictions
    hand = chunks[0].hand
    do_flip = hand == "left"
    left_mano = None
    if do_flip:
        left_mano = MANO(
            model_path="_DATA/data_left/mano_left",
            gender="neutral",
            num_hand_joints=15,
            create_body_pose=False,
            is_rhand=False,
        ).eval()
        with torch.no_grad():
            # Correct the mirrored left-hand shapedirs in the published MANO assets.
            left_mano.shapedirs[:, 0, :] *= -1
    print(f"Running HaWoR+MANO for {hand} hand: {len(chunks)} sequences ...")

    for batch_start in range(0, len(chunks), chunk_batch):
        group = chunks[batch_start : batch_start + chunk_batch]
        sequences: List[Dict[str, torch.Tensor]] = []
        for chunk in group:
            paths = np.asarray(
                [str(frame_path(frame_dir, int(index))) for index in chunk.frame_indices]
            )
            dataset = TrackDatasetEval(
                paths,
                chunk.boxes,
                img_focal=focal_length,
                img_center=image_center,
                normalization=True,
                dilate=1.2,
                do_flip=do_flip,
            )
            sequences.append(default_collate([dataset[index] for index in range(SEQUENCE_LENGTH)]))

        tensor_keys = [
            key for key, value in sequences[0].items() if isinstance(value, torch.Tensor)
        ]
        batch = {
            key: torch.stack([sequence[key] for sequence in sequences]).to(device)
            for key in tensor_keys
        }

        with torch.inference_mode():
            output = model(batch)
            model_output = output["out"]
            points_2d, _ = model.project(
                output["pred_keypoints_3d"],
                model_output["pred_cam"],
                batch["center"].flatten(0, 1),
                batch["scale"].flatten(0, 1),
                batch["img_focal"].flatten(0, 1),
                batch["img_center"].flatten(0, 1),
                return_full=True,
            )
            points_3d = output["pred_keypoints_3d"] + model_output["trans_full"]

        points_2d_cpu = points_2d.cpu()
        points_3d_cpu = points_3d.cpu()
        if do_flip:
            # HaWoR predicts a left hand from a horizontally flipped crop using
            # right-hand MANO. Undo the rotation convention and decode with the
            # actual left MANO layer, following the official video pipeline.
            rotations = model_output["pred_rotmat"].cpu()
            root_axis_angle = rotation_matrix_to_angle_axis(rotations[:, :1])
            pose_axis_angle = rotation_matrix_to_angle_axis(rotations[:, 1:])
            root_axis_angle[..., 1:] *= -1
            pose_axis_angle[..., 1:] *= -1
            left_rotations = torch.cat(
                [
                    angle_axis_to_rotation_matrix(root_axis_angle),
                    angle_axis_to_rotation_matrix(pose_axis_angle),
                ],
                dim=1,
            )
            assert left_mano is not None
            with torch.inference_mode():
                decoded = left_mano(
                    global_orient=left_rotations[:, :1],
                    hand_pose=left_rotations[:, 1:],
                    betas=model_output["pred_shape"].cpu(),
                    transl=model_output["trans_full"].cpu()[:, 0],
                    pose2rot=False,
                )
            points_3d_cpu = decoded.joints
            x = focal_length * points_3d_cpu[..., 0] / points_3d_cpu[..., 2]
            y = focal_length * points_3d_cpu[..., 1] / points_3d_cpu[..., 2]
            points_2d_cpu = torch.stack(
                [x + image_center[0], y + image_center[1]], dim=-1
            )

        points_2d_np = points_2d_cpu.reshape(
            len(group), SEQUENCE_LENGTH, 21, 2
        ).numpy()
        points_3d_np = points_3d_cpu.reshape(
            len(group), SEQUENCE_LENGTH, 21, 3
        ).numpy()
        del batch, output, model_output, points_2d, points_3d

        for sequence_index, chunk in enumerate(group):
            for offset in range(chunk.valid_count):
                frame_index = int(chunk.frame_indices[offset])
                predictions[frame_index] = {
                    "keypoints_2d": points_2d_np[sequence_index, offset].round(4).tolist(),
                    "keypoints_3d_camera": points_3d_np[sequence_index, offset].round(6).tolist(),
                }

        completed = min(batch_start + len(group), len(chunks))
        print(f"  {hand} inference: {completed}/{len(chunks)} sequences")

    return predictions


def build_result_document(
    video_path: Path,
    info: VideoInfo,
    focal_length: float,
    detection_frames: Sequence[Dict[str, Any]],
    segments_by_hand: Dict[str, List[TrackSegment]],
    predictions_by_hand: Dict[str, Dict[int, Dict[str, Any]]],
    hawor_root: Path,
) -> Dict[str, Any]:
    track_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {"left": {}, "right": {}}
    for hand, segments in segments_by_hand.items():
        for segment in segments:
            for position, frame_index_value in enumerate(segment.frame_indices):
                frame_index = int(frame_index_value)
                track_metadata[hand][frame_index] = {
                    "bbox_xyxy": segment.boxes[position].round(4).tolist(),
                    "bbox_interpolated": not bool(segment.detected[position]),
                    "detection_score": (
                        None
                        if np.isnan(segment.scores[position])
                        else round(float(segment.scores[position]), 6)
                    ),
                }

    frames: List[Dict[str, Any]] = []
    for frame_index in range(info.frame_count):
        hands: List[Dict[str, Any]] = []
        for hand in ("left", "right"):
            prediction = predictions_by_hand[hand].get(frame_index)
            track = track_metadata[hand].get(frame_index)
            if prediction is None or track is None:
                continue
            hands.append(
                {
                    "handedness": hand,
                    **track,
                    **prediction,
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / info.fps, 6),
                "hands": hands,
            }
        )

    detected_counts = {
        hand: sum(1 for frame in detection_frames if hand in frame.get("hands", {}))
        for hand in ("left", "right")
    }
    predicted_counts = {
        hand: len(predictions_by_hand[hand]) for hand in ("left", "right")
    }
    return {
        "metadata": {
            "source": str(video_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "HaWoR temporal prediction with MANO 21-joint decoding",
            "hawor_root": str(hawor_root),
            "coordinate_systems": {
                "keypoints_2d": "pixel coordinates [x, y] in the original frame",
                "keypoints_3d_camera": "camera-space MANO coordinates [x, y, z] in meters",
            },
            "joint_order": JOINT_NAMES,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "focal_length_pixels": focal_length,
            "image_center_pixels": [info.width / 2.0, info.height / 2.0],
            "detected_frame_counts": detected_counts,
            "predicted_frame_counts": predicted_counts,
        },
        "frames": frames,
    }


def render_hand_pose(frame: np.ndarray, hands: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Draw hand boxes and a 21-joint skeleton on a BGR video frame."""
    rendered = frame.copy()
    for hand in hands:
        handedness = hand["handedness"]
        color = HAND_COLORS[handedness]
        box = np.asarray(hand["bbox_xyxy"], dtype=np.int32)
        points = np.asarray(hand["keypoints_2d"], dtype=np.float32)
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        score = hand.get("detection_score")
        label = handedness.upper()
        if score is not None:
            label += f" {score:.2f}"
        cv2.putText(
            rendered,
            label,
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (15, 15, 15),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            label,
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        for edge_index, (start, end) in enumerate(SKELETON):
            finger_index = edge_index // 4
            start_point = tuple(np.rint(points[start]).astype(int))
            end_point = tuple(np.rint(points[end]).astype(int))
            cv2.line(
                rendered,
                start_point,
                end_point,
                FINGER_COLORS[finger_index],
                4,
                cv2.LINE_AA,
            )
        for joint_index, point in enumerate(points):
            center = tuple(np.rint(point).astype(int))
            radius = 7 if joint_index in (0, 4, 8, 12, 16, 20) else 5
            cv2.circle(rendered, center, radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(rendered, center, radius, color, -1, cv2.LINE_AA)
    return rendered


def render_video(
    input_path: Path,
    frame_dir: Path,
    result: Dict[str, Any],
    output_path: Path,
    info: VideoInfo,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the rendered video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.8f}",
        "-i", "pipe:0",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    print(f"Rendering {info.frame_count} frames to {output_path.name} ...")
    try:
        for frame_record in result["frames"]:
            frame_index = int(frame_record["frame_index"])
            frame = cv2.imread(str(frame_path(frame_dir, frame_index)))
            if frame is None:
                raise RuntimeError(f"cannot read cached frame {frame_index}")
            rendered = render_hand_pose(frame, frame_record["hands"])
            process.stdin.write(rendered.tobytes())
            if (frame_index + 1) % 500 == 0 or frame_index + 1 == info.frame_count:
                print(f"  rendering: {frame_index + 1}/{info.frame_count}")
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.stdin.close()
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_gap < 0:
        print("--max-gap must be non-negative", file=sys.stderr)
        return 2
    if args.chunk_batch < 1:
        print("--chunk-batch must be at least 1", file=sys.stderr)
        return 2
    if args.max_frames is not None and args.max_frames < 1:
        print("--max-frames must be at least 1", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.input.parent / f"{args.input.stem}_hand_pose"
    )
    frame_dir = output_dir / ".frame_cache"
    detection_cache = output_dir / "detections.json"
    keypoints_path = output_dir / f"{args.input.stem}_hand_keypoints.json"
    video_output = output_dir / f"{args.input.stem}_hand_pose.mp4"

    try:
        hawor_root = validate_hawor_root(args.hawor_root)
        device = select_device(args.device)
        info = read_video_info(args.input, args.max_frames)
        focal_length = args.focal_length or math.hypot(info.width, info.height)
        output_dir.mkdir(parents=True, exist_ok=True)

        cached = None if args.force_detect else compatible_detection_cache(
            detection_cache,
            frame_dir,
            args.input,
            info,
            args.det_confidence,
        )
        if cached is None:
            cached = detect_and_extract(
                video_path=args.input,
                info=info,
                frame_dir=frame_dir,
                cache_path=detection_cache,
                detector_path=hawor_root / "weights" / "external" / "detector.pt",
                confidence=args.det_confidence,
                device=device,
            )
        else:
            print(f"Reusing detection cache: {detection_cache}")

        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        segments_by_hand = {
            hand: build_track_segments(cached["frames"], hand, args.max_gap)
            for hand in ("left", "right")
        }
        chunks_by_hand = {
            hand: make_sequence_chunks(segments_by_hand[hand])
            for hand in ("left", "right")
        }

        with working_directory(hawor_root):
            model = load_hawor_model(hawor_root, device)
            predictions_by_hand = {
                hand: infer_hand(
                    model=model,
                    chunks=chunks_by_hand[hand],
                    frame_dir=frame_dir,
                    focal_length=focal_length,
                    image_center=(info.width / 2.0, info.height / 2.0),
                    device=device,
                    chunk_batch=args.chunk_batch,
                )
                for hand in ("left", "right")
            }

        result = build_result_document(
            video_path=args.input,
            info=info,
            focal_length=focal_length,
            detection_frames=cached["frames"],
            segments_by_hand=segments_by_hand,
            predictions_by_hand=predictions_by_hand,
            hawor_root=hawor_root,
        )
        atomic_json_write(result, keypoints_path)
        render_video(args.input, frame_dir, result, video_output, info)
        if not args.keep_cache:
            # Both artifacts are already on disk, so a failure to drop the
            # frame cache is a warning rather than a job failure.
            try:
                shutil.rmtree(frame_dir)
            except OSError as cleanup_error:
                print(
                    f"Warning: could not remove frame cache {frame_dir}: {cleanup_error}",
                    file=sys.stderr,
                )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Hand pose estimation failed: {error}", file=sys.stderr)
        return 1

    print(f"Keypoints: {keypoints_path}")
    print(f"Rendered video: {video_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
