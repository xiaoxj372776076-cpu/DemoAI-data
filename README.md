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
- [Image-text consistency operator](operators/image_text_consistency/README.md):
  score how well a local image matches one or more texts with a Chinese CLIP
  model.
- [Hand pose operator](operators/hand_pose/README.md): estimate both hands in a
  video with HaWoR, decode 21 MANO joints per hand, and render the skeleton onto
  a new video. Requires an external HaWoR checkout and MANO assets.
- [Document OCR operator](operators/doc_ocr/README.md): parse a document page
  (image, or one page of a PDF) into semantic HTML with dots.ocr, and write a
  structured JSON with per-block bbox, category and text.
- [PDF OCR operator](operators/pdf_ocr/README.md): the multi-page version of the
  document OCR operator — rasterise every page of a PDF, parse them with one
  loaded dots.ocr handle, and merge the results back into a single HTML document
  and JSON in the original page order. Every page must parse for the run to
  succeed.
