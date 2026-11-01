import time

import tiktoken
import torch

from ch05.plotting import plot_losses
from ch05.training import train_model_simple

from .alpaca_prompt import format_input
from .dataset import create_dataloaders
from .model import load_pretrained_gpt2
from .prepare_dataset import load_splits


def main():
    start_time = time.time()

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

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, tokenizer, device
    )

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
