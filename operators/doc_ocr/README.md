# doc_ocr — 文档图片 → HTML 结构化解析

把一个文档页面（图片或 PDF 的某一页）解析成**语义 HTML**，同时输出带版式坐标的结构化 JSON。

## 模型选型：dots.ocr（RedNote / 小红书 HiLab）

| 候选 | 结论 |
| --- | --- |
| **dots.ocr**（`rednote-hilab/dots.ocr`） | ✅ 采用。单模型端到端：版式检测 + 文字识别 + 表格（HTML）+ 公式（LaTeX）一次生成，OmniDocBench 上打平/超过 Gemini 2.5 Pro，1.7B 体量。本机用 MLX 量化版（`mlx-community/dots.ocr-6bit`）跑 Apple Silicon，无需 CUDA / vLLM。 |
| MinerU | ❌ 管线式（doclayout-yolo + UniMERNet + OCR），依赖多、装起来重，且重度依赖 CUDA / Paddle，本机 M1 不友好。 |
| PaddleOCR-VL | ❌ 需要 PaddlePaddle 运行时，macOS arm64 支持与体积都不占优。 |

dots.ocr 的官方 `prompt_layout_all_en` 正好产出我们要的东西：每个版式元素的 `bbox` + `category` + `text`，
其中表格输出 HTML、公式输出 LaTeX、其余输出 Markdown —— 本算子直接把它转成完整 HTML 文档。

> ⚠️ 实测（M1 + 6bit 量化）：**官方那段长提示词会被模型原样复述**，退化不出结果。
> 本算子改用等价但更短的紧凑版指令（`PROMPT_LAYOUT_ALL`），能稳定返回 JSON 列表；
> 两者规则一致（Table→HTML、Formula→LaTeX、其余→Markdown、按阅读顺序）。

## 安装

```bash
pip install -r operators/doc_ocr/requirements.txt   # mlx / mlx-vlm / pillow / pymupdf
```

权重默认用 `mlx-community/dots.ocr-6bit`（约 2.6 GB，首次运行自动下载）。
若要固定到已下载的本地目录：

```bash
export DEMOAI_DOC_OCR_MODEL=/Users/cc/Desktop/DemoAI-TrainingData/models/ocr/dots.ocr-6bit
# 或每次调用时 --model <路径>
```

> 模型下载走镜像：`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`

## 用法

```bash
# 1) 解析一张文档图片（生成 page.doc.html + page.doc.json）
.venv312/bin/python operators/doc_ocr/parse_document.py page.png

# 2) 直接吃 PDF：先把第 7 页按 200 DPI 转成图片再解析
.venv312/bin/python operators/doc_ocr/parse_document.py paper.pdf --pdf-page 7 --dpi 200

# 3) 顺手产出两张图：HTML 渲染图 + 版式框叠加图
.venv312/bin/python operators/doc_ocr/parse_document.py page.png --render --overlay

# 只做纯 OCR（不跑版式）时用 --prompt-mode ocr
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--output-dir` | 产物目录（默认放在输入文件旁） |
| `--model` | MLX 模型路径 / HF repo，覆盖 `DEMOAI_DOC_OCR_MODEL` |
| `--prompt-mode` | `layout_all`（默认）/ `layout_only` / `ocr` |
| `--max-tokens` | 生成上限（默认 8192，大表格页需要更多） |
| `--pdf-page` / `--dpi` | PDF 输入时转哪一页、多少 DPI |
| `--render` | 把生成的 HTML 光栅化成 `<stem>.doc.png` |
| `--overlay` | 把版式框画到原图上，输出 `<stem>.layout.png` |

## 产物

- `<stem>.doc.html` —— 语义 HTML：`Title → <h1>`、`Section-header → <h2>`、`Table → <table>`（模型直接给 HTML）、
  `Formula → <code>` 保留 LaTeX、页眉页脚/图注单独标注。
- `<stem>.doc.json` —— 结构化结果：`blocks[]`（每个元素带 `index` / `category` / `bbox` / `text_format` / `text`）、
  `category_counts`、来源与模型信息，便于下游按类别过滤或按坐标反查。
- `<stem>.doc.png`（`--render`）、`<stem>.layout.png`（`--overlay`）。

JSON 与 HTML 都走**原子写**（先 `.tmp` 再 `replace`），避免下游读到半截文件。

## 实现约定

- 与仓库其它算子一致：CLI + 结构化 JSON + 原子写文件；模型权重放 `~/Desktop/DemoAI-TrainingData`，不进仓库。
- 模型输出偶尔会被截断（长表格），`extract_json_objects()` 会兜底抢救已完整的 JSON 对象。
- 依赖都是懒加载：`--render` / `--overlay` / PDF 输入才需要 `pymupdf`、`pillow`。
