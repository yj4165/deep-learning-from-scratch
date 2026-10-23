import time

import tiktoken
import torch

from .dataset import create_dataloaders
from .metrics import calc_accuracy_loader
from .model import convert_to_classifier, load_pretrained_gpt2
from .plotting import plot_values
from .training import train_classifier_simple


def main():
    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    model = convert_to_classifier(load_pretrained_gpt2())
    model.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, test_loader = create_dataloaders(tokenizer)

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.1
    )
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50, eval_iter=5
        )
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs,
        label="accuracy"
    )

    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
