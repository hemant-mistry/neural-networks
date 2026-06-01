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
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        self.model = nn.Sequential(
            nn.Linear(image_dim+num_classes, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self,img,labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_embedding(labels)
        x = torch.cat((img_flat,c), dim=-1)
        validity = self.model(x)
        return validity
    
if __name__ == "__main__":
    # Load real dataset
    print("Downloading and loading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Dataset loaded successfully")


    # Training loop
    num_epochs = 50
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    print("--- Starting Training ---")

    for epoch in range(num_epochs):

        for i, (real_imgs, real_labels) in enumerate(dataloader):

            current_batch_size = real_imgs.size(0)

            valid_target = torch.ones(current_batch_size,1)
            fake_target = torch.zeros(current_batch_size,1)

            # Train Discriminator

            optimizer_D.zero_grad()

            real_preds = discriminator(real_imgs,real_labels)
            d_real_loss = criterion(real_preds, valid_target)

            z = torch.randn(current_batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (current_batch_size,))
            fake_imgs = generator(z, fake_labels)

            fake_preds = discriminator(fake_imgs.detach(), fake_labels)
            d_fake_loss = criterion(fake_preds, fake_target)

            d_loss = (d_real_loss+d_fake_loss)/2
            d_loss.backward()
            optimizer_D.step()


            # Train generator

            optimizer_G.zero_grad()

            g_preds = discriminator(fake_imgs, fake_labels)

            g_loss = criterion(g_preds, valid_target)

            g_loss.backward()
            optimizer_G.step()

            if i % 400 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                
    # Add this to the very bottom of number_generator.py
    torch.save(generator.state_dict(), "cgan_generator.pth")
    print("Model saved to cgan_generator.pth!")