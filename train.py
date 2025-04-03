import os
import torch
from model import GPT

def train_model(config, dataset):
    model = GPT(config)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):
        model.train()
        xb, yb = dataset.get_batch('train')
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % config.eval_interval == 0:
            print(f"Iter {iter}: Loss = {loss.item():.4f}")

    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")

if __name__ == "__main__":
    from config import Config
    from dataset import CharDataset, input_file_path

    config = Config()
    dataset = CharDataset(input_file_path, config)
    train_model(config, dataset)