# 🍔 fat_photo — добавь килограммы человеку на фото

Бесплатное приложение для Google Colab: загружаешь фото человека, крутишь
ползунок «сколько добавить кг» — и получаешь фото потолще. Полностью
бесплатно (без платных API): только open-source модели от Hugging Face.

## Как это работает

1. **MediaPipe** определяет ключевые точки лица (468 точек) и тела
   (33 точки позы) + маску человека.
2. **Геометрическая деформация** растягивает контуры лица (челюсть, щёки,
   двойной подбородок) и тела (торс, живот, бёдра, ляжки) пропорционально
   ползунку. Это работает мгновенно, без GPU.
3. **Опционально:** **Stable Diffusion img2img** (Realistic Vision 5.1)
   прогоняет результат с промптом *"obese, overweight person, double chin,
   chubby cheeks, large belly…"* и низкой силой диффузии (~0.25), чтобы
   убрать артефакты warp'а и добавить реалистичные складки кожи. Требует
   GPU — бесплатный T4 в Colab подходит.

Только силуэт деформируется внутри маски человека — фон остаётся нетронутым.

## Запуск в Google Colab (одной ячейкой)

1. Открой <https://colab.research.google.com> → *File → New notebook*.
2. *Runtime → Change runtime type → T4 GPU* (нужно для AI-улучшения; без него
   достаточно CPU).
3. Скопируй и запусти эту ячейку:

```python
import os, subprocess, sys, pathlib
REPO = 'https://github.com/GANGONai/TEXT-generator-ai.git'
DIR  = pathlib.Path('/content/TEXT-generator-ai')
if not DIR.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', REPO, str(DIR)])
else:
    subprocess.check_call(['git', '-C', str(DIR), 'pull', '--ff-only'])
os.chdir(DIR / 'fat_photo')
sys.path.insert(0, str(DIR))
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       '-r', 'requirements.txt'])

from fat_photo.app import launch
launch()  # печатает публичный *.gradio.live URL прямо в эту ячейку
```

Или открой готовый ноутбук: [`fat_photo_colab.ipynb`](./fat_photo_colab.ipynb).

Когда увидишь строку `Running on public URL: https://xxxx.gradio.live` —
открой её и пользуйся. Чтобы остановить — нажми ⏹ в Colab.

## Локальный запуск (Linux/Mac)

```bash
git clone https://github.com/GANGONai/TEXT-generator-ai.git
cd TEXT-generator-ai/fat_photo
pip install -r requirements.txt
python app.py
```

Откроется локальная страница (`http://127.0.0.1:7860`).

### Только геометрия, без AI

Если нужен **только** геометрический warp (нет GPU / не хочешь ждать
скачивания SD), просто не ставь галочку «AI-улучшение». В этом случае
зависимости `torch / diffusers / transformers` нужны только для импорта
проверок и MediaPipe; для самого warp'а достаточно `numpy + Pillow + gradio
+ mediapipe`.

## Структура

```
fat_photo/
├── README.md
├── app.py                    # Gradio UI + entrypoint (launch())
├── requirements.txt
├── fat_photo_colab.ipynb     # Colab-ноутбук с одной ячейкой запуска
└── src/
    ├── __init__.py
    ├── landmarks.py          # MediaPipe face / pose / mask
    ├── warp.py               # RBF backward-mapping warp на чистом NumPy
    ├── fatten.py             # построение контрольных точек для лица/тела
    ├── refine.py             # SD img2img (Realistic Vision 5.1)
    └── pipeline.py           # высокоуровневая функция fatten_photo()
```

## API

```python
from fat_photo.src.pipeline import fatten_photo

result = fatten_photo(
    "selfie.jpg",
    intensity=0.6,           # 0..1
    use_ai_refine=True,      # включить SD-уточнение
    ai_strength=0.25,        # сила диффузии, 0..1
    seed=42,                 # для воспроизводимости
)
result.image.save("fatter.png")
print(result.message)
```

## Что бесплатного используется

| Компонент              | Лицензия               | Где          |
|------------------------|------------------------|--------------|
| MediaPipe              | Apache 2.0             | pip-пакет    |
| Realistic Vision 5.1   | CreativeML OpenRAIL-M  | Hugging Face |
| Stable Diffusion 1.5   | CreativeML OpenRAIL-M  | Hugging Face |
| Gradio                 | Apache 2.0             | pip-пакет    |
| PyTorch / diffusers    | BSD / Apache           | pip-пакет    |

## Ограничения и заметки

- Лучший результат — на одном человеке в полный рост или поясной портрет,
  лицом к камере. Группы и сильные ракурсы пока обрабатываются плохо.
- Геометрический warp может «съесть» руки/предметы, если они близко к телу
  и попадают в зону влияния контрольных точек. Включи AI-улучшение —
  Stable Diffusion обычно дорисовывает.
- Без GPU AI-улучшение работает (на CPU), но 1 кадр займёт ~5–10 минут.
  С T4 — ~30–60 секунд.
- При очень больших значениях интенсивности (>80%) могут появляться
  артефакты в области шеи и под подбородком — это нормально.
