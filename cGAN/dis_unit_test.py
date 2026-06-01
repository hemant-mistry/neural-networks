import torch
import matplotlib.pyplot as plt
from number_generator import Generator, Discriminator

# 1. Instantiate both models
generator = Generator()
discriminator = Discriminator()

# 2. Create Dummy Data for the Generator
test_batch_size = 16 
dummy_noise = torch.randn(test_batch_size, 100) # latent_dim = 100
dummy_labels = torch.randint(0, 10, (test_batch_size,)) # labels 0-9

# --- TEST 1: GENERATOR ---
print("--- Testing Generator ---")
with torch.no_grad():
    generated_images = generator(dummy_noise, dummy_labels)
print(f"Generator Output Shape: {generated_images.shape}") 
# Expected: [16, 1, 28, 28]

# --- TEST 2: DISCRIMINATOR ---
print("\n--- Testing Discriminator ---")
# Let's pass the fake images we just made into the discriminator!
with torch.no_grad():
    predictions = discriminator(generated_images, dummy_labels)

print(f"Discriminator Output Shape: {predictions.shape}")
# Expected: [16, 1]

print("\nDiscriminator predictions for the first 5 fake images:")
for i in range(5):
    # .item() pulls the single number out of the PyTorch tensor
    prob = predictions[i].item() 
    print(f"Image {i+1}: {prob:.4f} (0=Fake, 1=Real)")