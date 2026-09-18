#!/usr/bin/env python3
"""Score image-text consistency with a CLIP model and write structured JSON."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_MODEL = "OFA-Sys/chinese-clip-vit-large-patch14"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "demoai-data" / "image-text-consistency"


def input_image(value: str) -> Path:
    """Validate and return an existing local image file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input image does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        raise argparse.ArgumentTypeError(f"unsupported image format: {suffix}")
    return path


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score how well a local image matches one or more texts with a "
            "Chinese CLIP model."
        )
    )
    parser.add_argument("input", type=input_image, help="local image file")
    parser.add_argument(
        "texts",
        nargs="+",
        help="one or more texts to score against the image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output file (default: INPUT.consistency.json beside the image)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face CLIP model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="model cache directory (default: ~/.cache/demoai-data/image-text-consistency)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=8,
        help="texts encoded per forward pass (default: 8)",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    """Return the default JSON path beside an input image."""
    return input_path.with_name(f"{input_path.stem}.consistency.json")


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


def json_value(value: Any) -> Any:
    """Convert model output values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"unsupported result value: {type(value).__name__}")


def prepare_environment(model_cache: Path) -> Path:
    """Create and return the Hugging Face cache directory for the model."""
    cache_path = model_cache.expanduser().resolve()
    hub_cache = cache_path / "hub"
    xet_cache = cache_path / "xet"
    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    return cache_path


def cosine_similarity_to_float(value: Any) -> float:
    """Convert a one-element tensor to a Python float."""
    return round(float(value.flatten()[0]), 6)


def score_texts(
    texts: Sequence[str],
    image_features,
    text_features,
) -> List[Dict[str, Any]]:
    """Return per-text cosine similarities and softmax probabilities."""
    import torch

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_norm @ text_norm.T).flatten()

    logit_scale = 100.0
    probabilities = torch.softmax(similarities * logit_scale, dim=0)

    results: List[Dict[str, Any]] = []
    for index, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "cosine_similarity": round(float(similarities[index]), 6),
                "consistency_score": round(float(probabilities[index]), 6),
            }
        )
    return results


def pooled_features(output: Any) -> Any:
    """Return the feature tensor regardless of the transformers return type."""
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output


def encode_texts(
    model,
    processor,
    texts: Sequence[str],
    device: str,
    batch_size: int,
):
    """Encode all texts in batches and concatenate the features."""
    import torch

    features: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_text_features(**inputs)
        features.append(pooled_features(outputs).cpu())
    return torch.cat(features, dim=0)


def encode_image(model, processor, image_path: Path, device: str):
    """Encode one image and return its pooled features on CPU."""
    import torch
    from PIL import Image

    with Image.open(str(image_path)) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    return pooled_features(features).cpu()


def score_image_texts(
    input_path: Path,
    texts: Sequence[str],
    model_name: str,
    model_cache: Path,
    device_request: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Run Chinese CLIP and return a JSON-compatible result document."""
    device = select_device(device_request)
    prepare_environment(model_cache)

    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = model.to(device).eval()

    image_features = encode_image(model, processor, input_path, device)
    text_features = encode_texts(model, processor, texts, device, batch_size)
    scores = score_texts(texts, image_features, text_features)

    best = max(scores, key=lambda item: item["consistency_score"])
    return {
        "source": str(input_path),
        "model": model_name,
        "device": device,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "best_match": {
            "text": best["text"],
            "consistency_score": best["consistency_score"],
        },
    }


def write_result(result: Dict[str, Any], output_path: Path) -> Path:
    """Atomically write a result document as UTF-8 JSON."""
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(args.input)
    )

    try:
        result = score_image_texts(
            input_path=args.input,
            texts=args.texts,
            model_name=args.model,
            model_cache=args.model_cache,
            device_request=args.device,
            batch_size=args.batch_size,
        )
        written_path = write_result(result, output_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Image-text consistency scoring failed: {error}", file=sys.stderr)
        return 1

    print(f"Scoring completed: {written_path}")
    for item in result["scores"]:
        print(
            f"  [{item['text']}] "
            f"consistency={item['consistency_score']:.4f} "
            f"cosine={item['cosine_similarity']:.4f}"
        )
    print(f"Best match: {result['best_match']['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
