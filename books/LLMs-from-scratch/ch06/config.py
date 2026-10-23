from pathlib import Path

from ch05.config import GPT2_BASE_CONFIG, GPT2_DIR, GPT2_MODEL_CONFIGS


CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {**GPT2_BASE_CONFIG, **GPT2_MODEL_CONFIGS[CHOOSE_MODEL]}

CH06_DIR = Path(__file__).resolve().parent
DATA_DIR = CH06_DIR / "data"
