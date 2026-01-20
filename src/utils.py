# utils

import torch
import torchvision.utils import save_image
import os

def save_samples(generator, epoch, noise_dim, device):
    z = torch.random(64, noise_dim).to(device)
    fake_images = generator(z)
    os.makedirs("outputs/images", exist_ok = True)
    save_image(fake_images, f"outputs/images/epoch_{epoch}.png", normalize = True)


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)