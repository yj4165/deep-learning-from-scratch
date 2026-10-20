from pathlib import Path


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # ch04 uses a context length of 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "model_and_optimizer.pth"
