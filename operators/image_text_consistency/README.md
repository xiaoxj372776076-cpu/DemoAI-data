# Image-text consistency operator

Scores how well a local image matches one or more texts with a Chinese CLIP
model. The default model is `OFA-Sys/chinese-clip-vit-large-patch14`, which
understands Chinese prompts natively.

For every input text the operator reports:

- `cosine_similarity`: cosine similarity between the image embedding and the
  text embedding, in the range [-1, 1];
- `consistency_score`: softmax probability across all supplied texts (higher
  means the text describes the image better relative to the other texts).

## Requirements

- Apple Silicon Mac (MPS), CUDA GPU, or CPU
- Python 3.10+
- Python dependencies from `requirements.txt`

```bash
python3 -m pip install -r operators/image_text_consistency/requirements.txt
```

## Usage

Score one image against two texts in a single run:

```bash
python3 operators/image_text_consistency/score_image_text.py \
  "/path/to/image.jpg" \
  "这是一个儿童" \
  "这是一个老虎"
```

The JSON result is written beside the input image by default as
`INPUT.consistency.json`. Choose an explicit output file when needed:

```bash
python3 operators/image_text_consistency/score_image_text.py \
  "/path/to/image.jpg" "一个孩子" \
  --output "/path/to/result.json"
```

Models are cached in `~/.cache/demoai-data/image-text-consistency` by default;
override that location with `--model-cache`. Use `--device` to pin the
inference device (`auto`, `cpu`, `mps`, or `cuda`) and `--model` to switch to
another Hugging Face CLIP-compatible model.
