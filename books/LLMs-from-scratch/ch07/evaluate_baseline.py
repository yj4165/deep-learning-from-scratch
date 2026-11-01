import tiktoken
import torch

from ch05.generation import generate, text_to_token_ids, token_ids_to_text
from ch05.losses import calc_loss_loader

from .alpaca_prompt import format_input
from .config import BASE_CONFIG
from .dataset import create_dataloaders
from .model import load_pretrained_gpt2
from .prepare_dataset import load_splits


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    train_data, val_data, test_data = load_splits()

    model = load_pretrained_gpt2()
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, tokenizer, device
    )

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=5
        )

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


if __name__ == "__main__":
    main()
