# training pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_samples, weights_init

batch_size = 128
epochs = 50
learning_rate = 0.002
noise_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)(0.5,))
])

dataset = datasets.MNIST(
    root = "data/mnist",
    train = True,
    download = True,
    transform = transform
)

dataloader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True
)

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()

optimizer_gen = torch.optim.Adam(
    generator.parameters(),
    lr = learning_rate,
    betas=(0.5, 0.999)
)

optimizer_dis = torch.optim.Adam(
    discriminator.parameters(),
    lr = learning_rate,
    betas = (0.5,0.999)
)

# Training Loop

for epoch in range (1, epochs+1):
    for real_images, _ in tqdm(dataloader, desc = f"Epoch {epoch}/{epochs}"):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        optimizer_dis.zero_grad()

        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_labels)

        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(noise)

        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_dis.step()

        optimizer_gen.zero_grad()

        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)

        g_loss.backward()
        optimizer_gen.step()

    save_samples(generator, epoch, noise_dim, device)
    print(f"Epoch [{epoch}/{epochs})] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

torch.save(generator.state_dict(), "outputs/checkpoints/generator.pth")
torch.save(discriminator.state_dict(), "outputs/checkpoints/discriminator.pth")
