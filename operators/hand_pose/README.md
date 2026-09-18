# HaWoR hand pose operator

Estimates both hands in a video with the official HaWoR pipeline:

1. detect left/right hand bounding boxes with HaWoR's YOLO detector;
2. run the 16-frame temporal HaWoR model;
3. decode the prediction with MANO into 21 joints;
4. write per-frame 2D and camera-space 3D keypoints to JSON;
5. draw bounding boxes and the 21-joint skeleton into an H.264 video.

Frames inside a short detection gap are assigned linearly interpolated bounding
boxes. Frames outside a detected track contain an empty `hands` array instead
of fabricated keypoints.

## Model assets

This operator uses an external checkout of the official
[HaWoR repository](https://github.com/ThunderVVV/HaWoR). The tested revision is
`66c7d4108d58a716deccd192cb7645170cdc7bd7`.

The checkout must contain:

```text
weights/external/detector.pt
weights/hawor/model_config.yaml
weights/hawor/checkpoints/hawor.ckpt
_DATA/data/mano/MANO_RIGHT.pkl
_DATA/data/mano_mean_params.npz
_DATA/data_left/mano_left/MANO_LEFT.pkl
```

`mano_mean_params.npz` ships with the MANO release alongside the two hand
models and is required by the left-hand MANO layer; without it inference fails
with `No such file or directory: '_DATA/data//mano_mean_params.npz'`.

Download HaWoR weights from the project's official release and obtain MANO
assets under the MANO license. Model assets are intentionally not committed to
this repository. HaWoR's published model is licensed CC BY-NC-ND 4.0 and is not
licensed for commercial use.

## Environment

Python 3.10+ and `ffmpeg` are required. Install dependencies in a virtual
environment:

```bash
python3 -m pip install -r operators/hand_pose/requirements.txt
```

On Apple Silicon the operator automatically uses Metal (`mps`); CUDA is used
when available, otherwise it falls back to CPU.

## Usage

The default HaWoR checkout is
`~/Desktop/DemoAI-TrainingData/models/hawor/runtime`. Outputs are written next
to the input under an `INPUT_STEM_hand_pose` directory.

```bash
python3 operators/hand_pose/estimate_hands.py \
  ~/Desktop/DemoAI-TrainingData/left_cam.mp4
```

Use an explicit model or output location when needed:

```bash
python3 operators/hand_pose/estimate_hands.py INPUT.mp4 \
  --hawor-root /path/to/HaWoR \
  --output-dir ~/Desktop/DemoAI-TrainingData/hand_pose/run-001
```

`--max-frames 64 --keep-cache` is useful for a quick visual validation before a
full run. A compatible detection/frame cache is reused after an interrupted
run; pass `--force-detect` to rebuild it.

## Output schema

The JSON contains one record for every source frame. Each hand record includes
`handedness`, `bbox_xyxy`, `bbox_interpolated`, `detection_score`, 21
`keypoints_2d` pixel coordinates, and 21 `keypoints_3d_camera` MANO coordinates.
The exact joint order is recorded in `metadata.joint_order`.
