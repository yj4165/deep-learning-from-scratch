import tiktoken
import torch

from ch04.gpt_model import GPTModel, generate_text_simple
from ch05.config import GPT_CONFIG_124M
from ch05.generation import generate, text_to_token_ids, token_ids_to_text


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    encoded = text_to_token_ids("Every effort moves you", tokenizer)
    token_ids = generate_text_simple(
        model=model,
        idx=encoded,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Greedy output:\n", token_ids_to_text(token_ids, tokenizer))

    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Sampled output:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
