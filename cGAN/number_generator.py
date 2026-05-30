import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# -- Hyperparameters --
latent_dim = 100
num_classes = 10
image_dim = 28 * 28
batch_size = 64
learning_rate = 0.0002


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        # Coverts 0-9 to 10 dimension vectors
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        # Neural network 
        self.model = nn.Sequential(
            nn.Linear(latent_dim+num_classes, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512,image_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c  = self.label_embedding(labels)
        x = torch.cat((noise,c), dim=-1)
        img = self.model(x)

        # Fold the flat 784 into 28 * 28
        img = img.view(img.size(0), 1, 28, 28)
        return img
