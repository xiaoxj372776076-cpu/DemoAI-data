# Hugging Face dataset downloader

Downloads a complete Hugging Face dataset repository snapshot from either a
dataset ID or a Hub URL. Files keep the same directory structure used on the
Hub, and interrupted downloads can be resumed by running the command again.

## Requirements

- Python 3.10+
- Python dependencies from `requirements.txt`

```bash
python3 -m pip install -r crawlers/huggingface/requirements.txt
```

## Usage

Pass a dataset ID:

```bash
python3 crawlers/huggingface/download_dataset.py stanfordnlp/imdb
```

Or pass its Hugging Face URL:

```bash
python3 crawlers/huggingface/download_dataset.py \
  https://huggingface.co/datasets/stanfordnlp/imdb
```

Datasets are saved under `~/Downloads/huggingface-datasets` by default. Set a
different parent directory with `--output-dir`:

```bash
python3 crawlers/huggingface/download_dataset.py stanfordnlp/imdb \
  --output-dir /tmp/datasets
```

Download metadata and resumable-transfer state are stored in a hidden `.cache`
directory beside the downloaded datasets.

Use `--revision` to download a branch, tag, or commit other than the repository
default. Use `--force` to download files again instead of reusing matching local
files.

For private or gated datasets, authenticate first with `hf auth login`. Tokens
are managed by the Hugging Face client and should not be added to this project.
