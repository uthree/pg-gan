import os
import multiprocessing
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)

class Blur(nn.Module):
    """Some Information about Blur"""
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, upsample=True):
        super(GeneratorBlock, self).__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)
        else:
            self.upsample = nn.Identity()

        self.conv1 = nn.Conv2d(input_channels, latent_channels, 3, 1, 1, padding_mode='replicate')
        self.conv1.bias.data = torch.zeros(*self.conv1.bias.data.shape)
        self.act1 = leaky_relu()
        self.conv2 = nn.Conv2d(latent_channels, output_channels, 3, 1, 1, padding_mode='replicate')
        self.conv2.bias.data = torch.zeros(*self.conv2.bias.data.shape)
        self.act2 = leaky_relu()
        self.to_rgb = nn.Conv2d(output_channels, 3, 1, 1, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        rgb = self.to_rgb(x)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, initial_channels):
        super(Generator, self).__init__()
        self.last_channels = initial_channels
        self.layers = nn.ModuleList([])
        self.alpha = 0
        self.blur = Blur()
        self.latent2pic = nn.Linear(initial_channels, initial_channels * 4 * 4)
        self.upsample = nn.Upsample(scale_factor=2)
        self.tanh = nn.Tanh()
        
        self.add_layer(initial_channels,  False)

    def forward(self, x):
        # expected x shape is [N, initial_channels]
        rgb_out = None
        num_layers = len(self.layers)
        x = self.latent2pic(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        for i in range(len(self.layers)):
            x, rgb = self.layers[i](x)
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = self.blur(self.upsample(rgb_out)) + rgb
        rgb_out = self.tanh(rgb_out)
        return rgb_out

    def add_layer(self, channels, upsample=True):
        latent_channels = (channels + self.last_channels) // 2
        self.layers.append(GeneratorBlock(self.last_channels, latent_channels, channels, upsample=upsample))
        self.last_channels = channels


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, output_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()

        self.from_rgb = nn.Conv2d(3, input_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(input_channels, latent_channels, 3, 1, 1, padding_mode='replicate')
        self.act1 = leaky_relu()
        self.conv2 = nn.Conv2d(latent_channels, output_channels, 3, 1, 1, padding_mode='replicate')
        self.act2 = leaky_relu()
        self.res = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        if downsample:
            self.downsample = nn.Conv2d(output_channels, output_channels, 2, 2, 0)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act1(x) + r
        x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, initial_channels):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([])
        self.last_channels = initial_channels
        self.fc1 = nn.Linear(initial_channels+1, 64) 
        self.act1 = leaky_relu()
        self.fc2 = nn.Linear(64, 1)
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.blur = Blur()
        self.alpha = 0
        
        self.add_layer(initial_channels, downsample=False)

    def forward(self, rgb):
        x = self.layers[0].from_rgb(rgb)
        for i in range(len(self.layers)):
            if i == 1:
                x = x * self.alpha + self.layers[1].from_rgb(self.downsample(self.blur(rgb))) * (1-self.alpha)
            x = self.layers[i](x)
        minibatch_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
        x = x.mean(dim=[2,3])
        x = x.view(x.shape[0], -1)
        x = self.fc1(torch.cat([x, minibatch_std], dim=1))
        x = self.act1(x)
        x = self.fc2(x)
        return x
        
    def add_layer(self, channels, downsample=True):
        latent_channels = (channels + self.last_channels) // 2
        self.layers.insert(0, DiscriminatorBlock(channels, latent_channels, self.last_channels, downsample=downsample))
        self.last_channels = channels

def write_image(image, path):
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 127.5 + 127.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image, mode='RGB')
    image.save(path)

class GAN(nn.Module):
    def __init__(self, initial_channels=512, max_resolution=1024, min_channels=12, min_batch_size=4):
        super(GAN, self).__init__()
        self.initial_channels = initial_channels
        self.max_resolution = max_resolution
        self.min_channels = min_channels
        self.min_batch_size = min_batch_size
        
        self.generator = Generator(initial_channels)
        self.discriminator = Discriminator(initial_channels)
        
    def train_resolution(self, dataset, num_epoch=1, batch_size=4, result_dir ='results/', model_path='model.pt', lr=1e-4, device=torch.device('cpu'), augmentation=nn.Identity()):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr)
        bar = tqdm(total=num_epoch)
        self.to(device)
        D = self.discriminator
        G = self.generator
        T = augmentation
        for epoch in range(num_epoch):
            for i, real in enumerate(dataloader):
                alpha = epoch / num_epoch
                if len(G.layers) == 1:
                    alpha = 1
                G.alpha = alpha
                D.alpha = alpha

                real = real.to(device)
                N = real.shape[0]
                # train generator
                G.zero_grad()
                z = torch.randn(N, self.initial_channels, device=device)
                fake = G(z)
                generator_loss = -D(fake).mean()
                generator_loss.backward()
                optimizer_g.step()

                # train disciriminator
                D.zero_grad()
                discriminator_loss_fake = -torch.minimum(-D(T(fake.detach())) - 1, torch.zeros(N, 1).to(device)).mean()
                discriminator_loss_real = -torch.minimum(D(T(real)) - 1, torch.zeros(N, 1).to(device)).mean()
                discriminator_loss = discriminator_loss_fake + discriminator_loss_real
                discriminator_loss.backward()
                optimizer_d.step()

                bar.set_description(desc=f"Dloss: {discriminator_loss.item():.4f}, Gloss: {generator_loss.item():.4f} alpha: {alpha:.4f}")

            # write result
            write_image(fake[0], os.path.join(result_dir, f"{epoch}.png"))
            bar.update(1)
            torch.save(self, model_path)

    def train(self, dataset, num_epoch=1, batch_size=4, lr=1e-4, device=None, result_dir='./results/', model_path = './model.pt', augmentation=nn.Identity()):
        if device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        while True:
            image_size = 4 * 2 ** (len(self.generator.layers)-1)
            bs = batch_size // 2 ** (len(self.generator.layers)-1)
            ch = self.initial_channels // 2 ** (len(self.generator.layers)-1)

            if bs < self.min_batch_size:
                bs = self.min_batch_size
            if ch < self.min_channels:
                ch = self.min_channels

            dataset.set_size(image_size)
            self.train_resolution(dataset, num_epoch, bs, result_dir, model_path, lr, device, augmentation)
            
            if image_size >= self.max_resolution:
                break
            self.generator.add_layer(ch)
            self.discriminator.add_layer(ch)
            self.to(device)
