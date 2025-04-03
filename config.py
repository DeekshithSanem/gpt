# ----------------------------
# Config (Hyperparameters)
import torch
# ----------------------------
class Config:
    block_size = 128       # context window
    vocab_size = None      # to be set after dataset is processed
    n_embd = 128           # embedding dimension
    n_head = 4             # number of attention heads
    n_layer = 4            # number of transformer blocks
    dropout = 0.1
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 100
    #device = 'gpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # or 'cpu'
    seed = 42
    model_save_path = 'checkpoint/gpt_model.pt'