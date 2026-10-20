import torch

from ch04.gpt_model import GPTModel
from ch05.config import GPT_CONFIG_124M


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345],
                            [1107, 588, 11311]])

    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)

    target_probas_1 = probas[0, [0, 1, 2], targets[0]]
    target_probas_2 = probas[1, [0, 1, 2], targets[1]]
    print("Text 1:", target_probas_1)
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    neg_avg_log_probas = -torch.mean(log_probas)
    print("Negative average log probability:", neg_avg_log_probas)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Logits shape:", logits_flat.shape)
    print("Targets shape:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Cross-entropy loss:", loss)


if __name__ == "__main__":
    main()
