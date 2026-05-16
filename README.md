# 🎶 Song Lyrics AI Studio

Free, open-source **song-lyrics generator** that you can fine-tune on your own
songs. Runs entirely in Google Colab (or locally) using only free
Hugging Face models — no paid APIs, no closed services.

![banner](./monstr-13-0.png)

## ✨ Features

- 🚀 **One-cell Colab launch** — clone, install, run, share a public Gradio link.
- 🤖 **Multiple base models** — DistilGPT-2, GPT-2, GPT-2 medium, ruGPT-3 small,
  TinyLlama 1.1B (all free Hugging Face checkpoints).
- 🎓 **Fine-tune on your own lyrics** with a custom training loop:
  - per-step progress + ETA in the UI,
  - graceful OOM recovery (auto-shrinks batch size),
  - mixed-precision (fp16) on GPU, automatic CPU fallback,
  - gradient checkpointing to save VRAM.
- 📝 **Three tabs** in the web UI:
  1. **Generator** — pick a model, set title / theme / genre / language, tweak
     `temperature` / `top_p` / `top_k` / `repetition_penalty`, copy or download
     the result as `.txt`.
  2. **Upload lyrics** — manage the per-model dataset: add / edit / delete songs,
     full-text search, bulk import from `.txt` / `.csv`, kick off training and
     watch progress live.
  3. **Models** — create / delete models, **import** and **export** them as
     ZIP archives that contain weights, tokenizer, settings *and* the song
     dataset.
- 🇷🇺 / 🇬🇧 Russian + English support out of the box (auto-detected from the prompt).
- 📜 **Generation history** per model (last 200 results).
- 💾 **Auto-save** of songs, history and model metadata on every change.
- 🪵 **Rotating file logs** under `data/logs/app.log`.

## 📦 Project structure

```
TEXT-generator-ai/
├── app.py                     # CLI / Colab entrypoint (one-cell launch)
├── requirements.txt
├── notebooks/colab_run.ipynb  # ready-to-run Colab notebook
└── src/
    ├── config.py              # base-model registry, defaults, paths
    ├── logger.py              # rotating logger
    ├── utils.py               # slugs, JSON helpers, text utils
    ├── models/                # ModelManager (CRUD, weights I/O)
    ├── datasets/              # SongManager (per-model dataset)
    ├── training/              # LyricsTrainer (fine-tuning loop)
    ├── generation/            # LyricsGenerator + HistoryStore
    ├── exports/               # ZIP import / export
    └── ui/                    # Gradio Blocks app + three tabs
```

Runtime data lives under `data/` (gitignored):

```
data/
├── models/<slug>/             # one folder per user model
│   ├── config.json            #   metadata
│   ├── songs.json             #   user's lyrics dataset
│   └── weights/               #   HF weights + tokenizer (after training)
├── history/<slug>.json        # generation history
├── exports/                   # generated ZIPs
└── logs/                      # rotating logs
```

## 🚀 Running in Google Colab

The fastest path — **one cell**:

1. Open <https://colab.research.google.com> → *File → New notebook*.
2. (Recommended) `Runtime → Change runtime type → T4 GPU`.
3. Paste this single cell and run it:

```python
import os, subprocess, sys, pathlib
REPO = 'https://github.com/GANGONai/TEXT-generator-ai.git'
DIR  = pathlib.Path('/content/TEXT-generator-ai')
if not DIR.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', REPO, str(DIR)])
else:
    subprocess.check_call(['git', '-C', str(DIR), 'pull', '--ff-only'])
os.chdir(DIR)
sys.path.insert(0, str(DIR))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

# Import and launch directly — the public *.gradio.live URL prints
# into THIS cell's output (using subprocess would buffer it).
from app import launch
launch()  # auto-detects Colab → share=True, inbrowser=False
```

The cell keeps running while the server is up — that is normal. Look for a
line like `Running on public URL: https://xxxx.gradio.live` and open it.

To stop the app, click the ⏹ *interrupt* button in Colab. To re-launch later,
re-run the cell.

Or open the ready-made notebook: [`notebooks/colab_run.ipynb`](./notebooks/colab_run.ipynb).

### Persisting data across Colab sessions

Mount Google Drive and point the data dir at it:

```python
from google.colab import drive
drive.mount('/content/drive')
os.environ['LYRICS_AI_DATA'] = '/content/drive/MyDrive/lyrics-ai'
```

Now your models, songs and generation history survive runtime restarts.

## 💻 Running locally

```bash
git clone https://github.com/GANGONai/TEXT-generator-ai.git
cd TEXT-generator-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open <http://localhost:7860>.

You can also import `launch()` from any Python session / notebook:

```python
from app import launch
launch()                 # local server on port 7860
launch(share=True)       # also expose a public *.gradio.live URL
```

CLI flags:

| Flag           | Default   | Purpose                                  |
|----------------|-----------|------------------------------------------|
| `--share`      | `false`   | Expose a public `*.gradio.live` URL (auto-on in Colab). |
| `--port`       | `7860`    | Local port.                              |
| `--host`       | `0.0.0.0` | Bind host.                               |
| `--no-browser` | `false`   | Don't open a browser automatically.      |
| `--debug`      | `false`   | Verbose Gradio logs.                     |

Environment variables: `LYRICS_AI_DATA`, `LYRICS_AI_PORT`, `LYRICS_AI_HOST`,
`LYRICS_AI_SHARE`.

## 🧪 Quick usage walkthrough

1. **Models tab** → *Create model* → pick a name + base model (e.g. DistilGPT-2)
   → *Create*.
2. **Upload lyrics tab** → select the model → paste a song into the right pane
   → *Save song*. Repeat for as many songs as you want (or use *Import* to load
   a `.txt`/`.csv`). 20–50 songs is usually enough for a noticeable style.

   Bulk-import TXT layout (UTF-8). Songs are separated by **two blank lines**,
   each block may start with optional `Название:` / `Жанр:` / `Текст:` headers
   (or their English equivalents `Title:` / `Genre:` / `Lyrics:`):

   ```
   Название: Ночной город
   Жанр: рок
   Текст:
   Я иду по улице ночной
   Город спит, но не со мной

   Свети, моя звезда
   Не гасни никогда


   Название: Морская
   Жанр: шансон
   Море, море, мир бездонный
   Долгий путь и тёмный путь
   ```

   Headers are optional — a file with just blocks of text is imported as
   "Песня #1", "#2", … For CSV use a UTF-8 file with `text` (required), and
   `title` / `genre` (optional) columns.
3. Still in *Upload lyrics* → set epochs (3 is a sane default) → *Start training*.
   Watch progress in the status box; ETA and percent update in real time.
4. **Generator tab** → select the trained model → enter a title / theme →
   *Generate*. Tweak `temperature` / `top_p` for more or less creative output.
5. **Models tab** → *Export model* to download a ZIP archive containing the
   weights, tokenizer, settings and dataset. *Import model* re-creates the
   model on any other machine.

## 🛠 Tech stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for
  model loading, tokenizers and generation.
- [PyTorch](https://pytorch.org/) for fine-tuning.
- [Gradio 4](https://www.gradio.app/) for the web UI.
- 100 % standard library for the JSON-based dataset, ZIP I/O and logger —
  no external database, no API keys.

## ⚠️ Hardware notes

- Most things work on CPU but training is slow. A free Colab **T4 GPU** is the
  sweet spot for `distilgpt2` / `gpt2` / `rugpt3small`.
- `gpt2-medium` and `tinyllama-1.1b` require a GPU. The trainer auto-enables
  fp16 and gradient checkpointing, and will halve the batch size if it sees
  CUDA OOM.

## 📄 License

MIT.  All bundled models are downloaded directly from the Hugging Face Hub
and retain their original licenses.
