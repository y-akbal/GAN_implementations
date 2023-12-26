## Below we implement a simple GAN for tabular data generation for experimental purposes!!!
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class training_setup:
    z_dim: str
    latent_dim: float
    num_features:int
    data_set: str  
    lr:float = 1e-4
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size:int = 32 
    num_epochs:int = 50
    seed: int = 0

class Discriminator(nn.Module):
    def __init__(self, 
                 in_features:int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, 
                 input_dim:int = 128, 
                 latent_dim:int = 8):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, latent_dim),
            nn.Tanh(), ##Squash the stuff into (-1,1) (do we really need this???) 
        )
    def forward(self, x):
        return self.generator(x)


def return_disc_gen(setup:dataclass = training_setup(10, 10, 10, "w")):
    ## Grab the attributes of the
    num_features, z_dim, device, batch_size = setup.num_features, setup.z_dim, setup.device, setup.batch_size
    ## Disc and gen
    disc = Discriminator(num_features).to(device)
    gen = Generator(z_dim, num_features).to(device)
    ## Fixing the seed is needed for reproducibility...
    torch.manual_seed(setup.seed)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    return disc, gen, fixed_noise

def prepare_data(setup: dataclass, 
                 numerical_features:list[int],
                categorical_features:list[int]):
    pd_data = pd.read_csv(setup.data_dir)
    ## Do some preprocessing here!!!
    ## Normalize the data --- convert to tensor---
    ## May wish to use the collating function 
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
    ##Here training will happen here, and we will save the model at the end
    return None


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
