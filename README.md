# DemoAI-data

Data processing pipelines for DemoAI

## Crawlers

- [YouTube downloader](crawlers/youtube/README.md): download a video by its
  YouTube ID, capped at 1080p and 60 fps.
- [Hugging Face dataset downloader](crawlers/huggingface/README.md): download a
  complete dataset snapshot by Hub URL or dataset ID.

## Operators

- [ASR operator](operators/asr/README.md): transcribe local video or audio with
  an open-source Whisper model and write timestamped JSON output.
