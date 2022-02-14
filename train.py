from model import GAN
from dataset import ImageDataset
import torch
import os
import sys

ds = ImageDataset(sys.argv[1:], max_len=5000)
if os.path.exists('model.pt'):
    model = torch.load('model.pt')
    print("Loaded model")
else:
    print("Creating new model...")
    model = GAN()
    print("Created new model")
model.train(ds, num_epoch=200, batch_size=256, lr=1e-5)
