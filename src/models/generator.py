# Generator architecture 

import torch
import torch.nn as nn

# Define a simple Generator model

class Generator(nn.Module):
    # Generator model to generate images from random noise
    # Input: random noise vector z
    # Output: fake MNIST image (1x28x28)

    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.linear(noise_dim, 256),
            nn.ReLU(True),
            nn.linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self,z):
        img = self.model(z)
        return img.view(-1,1,28,28)