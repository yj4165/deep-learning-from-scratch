import torch

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


def convert_to_classifier(model, num_classes=2):
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    # Train only the new head, the last transformer block and the final norm
    for param in model.parameters():
        param.requires_grad = False

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    for param in model.out_head.parameters():
        param.requires_grad = True

    return model
