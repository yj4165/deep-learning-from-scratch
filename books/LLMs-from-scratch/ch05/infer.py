import tiktoken
import torch

from ch04.gpt_model import GPTModel

from .config import CHECKPOINT_PATH, GPT_CONFIG_124M
from .generation import generate, text_to_token_ids, token_ids_to_text


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint.get("model_config", GPT_CONFIG_124M)
    model = GPTModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=15,
        context_size=model_config["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
