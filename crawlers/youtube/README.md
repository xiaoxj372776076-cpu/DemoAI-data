# YouTube downloader

Downloads a YouTube video by video ID. The downloader selects the highest
available quality that does not exceed 1080p or 60 fps, then merges video and
audio into an MP4 file.

Only download videos when you have permission from the copyright owner and the
download complies with YouTube's terms and applicable law.

## Requirements

- Python 3.10+
- `ffmpeg` available on `PATH`
- A supported JavaScript runtime such as Node.js or Deno
- Python dependencies from `requirements.txt`

```bash
python3 -m pip install -r crawlers/youtube/requirements.txt
```

## Usage

Downloads are saved to `~/Downloads` by default:

```bash
python3 crawlers/youtube/download_video.py -- -UkD8BG_9nU
```

To use another destination:

```bash
python3 crawlers/youtube/download_video.py --output-dir /tmp/videos -- -UkD8BG_9nU
```

The `--` separator is needed only when a video ID starts with `-`.

If YouTube asks you to sign in to confirm you are not a bot, reuse an existing
browser session without exporting cookies to a file:

```bash
python3 crawlers/youtube/download_video.py --cookies-from-browser chrome -- -UkD8BG_9nU
```

Supported browser values are `chrome`, `safari`, `firefox`, `edge`, and `brave`.
Browser cookies are read by `yt-dlp` at runtime and are never stored by this
project.
