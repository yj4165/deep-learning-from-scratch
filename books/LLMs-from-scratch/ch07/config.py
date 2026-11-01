from pathlib import Path

from ch05.config import GPT2_BASE_CONFIG, GPT2_DIR, GPT2_MODEL_CONFIGS


CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG = {**GPT2_BASE_CONFIG, **GPT2_MODEL_CONFIGS[CHOOSE_MODEL]}

CH07_DIR = Path(__file__).resolve().parent
DATA_DIR = CH07_DIR / "data"
DATA_FILE_PATH = DATA_DIR / "instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
