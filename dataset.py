import os
import requests
import torch

# Auto-download tinyshakespeare if not present
input_file_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare.txt')
os.makedirs(os.path.dirname(input_file_path), exist_ok=True)

if not os.path.exists(input_file_path):
    print("Downloading tinyshakespeare dataset...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

class CharDataset:
    def __init__(self, file_path, config):
        self.config = config
        self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.config.vocab_size = self.vocab_size
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.config.block_size

    def get_batch(self, split='train'):
        ix = torch.randint(len(self), (self.config.batch_size,))
        x = torch.stack([self.data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.config.block_size+1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)

__all__ = ['CharDataset', 'input_file_path']