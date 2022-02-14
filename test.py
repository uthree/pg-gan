from model import GAN
from dataset import ImageDataset
import torch
import os
import sys
from model import write_image


if os.path.exists('model.pt'):
    model = torch.load('model.pt')
    print("Loaded model")
else:
    print("Creating new model...")
    model = GAN()
    print("Created new model")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists('./tests/'):
    os.mkdir('./tests/')

num_images = int(sys.argv[1])
model = model.to(device)
for i in range(num_images):
    z = torch.randn(1, model.initial_channels, device=device)
    img = model.generator(z)
    path = os.path.join('./tests/', f"{i}.png")
    write_image(img[0], path)
