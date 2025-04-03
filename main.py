from config import Config
from dataset import CharDataset
from train import train_model

config = Config()
dataset = CharDataset("data/shakespeare.txt", config)  # make sure this file exists
train_model(config, dataset)
