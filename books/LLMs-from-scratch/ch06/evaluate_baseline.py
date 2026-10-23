import tiktoken
import torch

from .dataset import create_dataloaders
from .metrics import calc_accuracy_loader
from .model import convert_to_classifier, load_pretrained_gpt2


def main():
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
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device, num_batches=10
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
