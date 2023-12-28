## Below we implement a simple GAN for tabular data generation for experimental purposes!!!
import pandas as pd
import numpy as np
import tqdm as tqdm
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class training_setup:
    z_dim: int  = 128
    latent_dim: int = 256
    num_features:int = 10
    data_set: str  = os.path.join("datasets", "dataset.csv")
    lr:float = 1e-4
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size:int = 32 
    num_epochs:int = 50
    seed: int = 0
    save_dir: str = "generator_model"

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
    ## Grab the attributes
    num_features, z_dim, device = setup.num_features, setup.z_dim, setup.device
    ## Disc and gen, 
    disc = Discriminator(num_features).to(device)
    gen = Generator(z_dim, num_features).to(device)
    return disc, gen

def prepare_data(setup: dataclass, 
                numerical_features:list[int],
                categorical_features:list[int],
                split = True):
    data = pd.read_csv(setup.data_dir)
    data_x = data.iloc[:, numerical_features].to_numpy()  #RxF -- Rows x Features -> numpy_array
    data_y = data.iloc[:, categorical_features].to_numpy()

    normalized_data_x = (data_x - data_x.min(axis = 1))/(data_x.max(axis = 1) - data_x.min(axis = 1))
    if split:
        return train_test_split((normalized_data_x, data_y)) ## train_x, train_y, test_x, test_y splitted
    return normalized_data_x, data_y

def data_pipe_line(setup, data:tuple(np.ndarray, np.ndarray)):
    return DataLoader(data, batch_size=setup.batch_size, shuffle=True)

def train(setup: dataclass, 
          training_data: torch.Tensor,
          discriminator: nn.Module, 
          generator: nn.Module):
    ## 
    device = setup.device
    lr = setup.lr
    num_epochs = setup.num_epochs
    z_dim = setup.z_dim
    ##
    
    opt_disc = optim.Adam(discriminator.parameters(), lr = lr)
    opt_gen = optim.Adam(generator.parameters(), lr = lr)
    loss_fn = nn.BCELoss()
    ## Fixing the seed is needed for reproducibility...
    torch.manual_seed(setup.seed)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    for epoch in range(setup.num_epochs):
        for batch_idx, (real, _) in enumerate(training_data):
            real = real.to(setup.device)
            batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = generator(noise)
            disc_real = discriminator(real)
            lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake)
            lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            discriminator.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = discriminator(fake).view(-1)
            lossG = loss_fn(output, torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                 print(
                     f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                     Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                     )

            with torch.no_grad():
                fake = generator(fixed_noise)
                ### do your magic here!!!!
                ### logging some loss or something maybe some comparison would be good!!! 
                ### things like AE would be good for projection to two dimensional space.
    try:
        torch.save(generator.cpu(), "generator_weights.w")
        print("Generator saved!!!!")

    except Exception as e:
        print(f"Some thing went wrong with {e}")
    return None

def main():
    ### This dude is the main function we shall be using!!!
    return None

if __name__ == "__main__":
    
    main()
