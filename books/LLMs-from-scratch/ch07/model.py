from ch04.gpt_model import GPTModel
from ch05.gpt_download import download_and_load_gpt2
from ch05.load_weights_into_gpt import load_weights_into_gpt

from .config import BASE_CONFIG, CHOOSE_MODEL, GPT2_DIR


def load_pretrained_gpt2():
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=GPT2_DIR
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    return model
