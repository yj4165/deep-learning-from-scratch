import tiktoken
import torch

from ch04.gpt_model import GPTModel
from ch05.config import GPT2_BASE_CONFIG, GPT2_DIR, GPT2_MODEL_CONFIGS
from ch05.generation import generate, text_to_token_ids, token_ids_to_text
from ch05.gpt_download import download_and_load_gpt2
from ch05.load_weights_into_gpt import load_weights_into_gpt


def main():
    model_name = "gpt2-small (124M)"
    model_config = {**GPT2_BASE_CONFIG, **GPT2_MODEL_CONFIGS[model_name]}
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=GPT2_DIR
    )
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())

    model = GPTModel(model_config)
    load_weights_into_gpt(model, params)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=model_config["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
