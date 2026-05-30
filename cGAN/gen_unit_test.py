import torch
import matplotlib.pyplot as plt
from number_generator import Generator


generator = Generator()
latent_dim = 100
test_batch_size = 16

dummy_noise = torch.rand(test_batch_size, latent_dim)

dummy_labels = torch.randint(0, 10,(test_batch_size,))

print("--- Running Forward Pass ---")
print(f"Input Noise Shape: {dummy_noise.shape}")
print(f"Input Labels Shape: {dummy_labels.shape}")

with torch.no_grad():
    generated_images = generator(dummy_noise, dummy_labels)

print(f"Output Image Shape: {generated_images.shape}")
print("Success! The math works.")

# 4. Let's visualize what an untrained Generator draws!
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    
    # Grab one image, convert from PyTorch Tensor to NumPy array
    img_array = generated_images[i].numpy()
    
    # PyTorch images are (Channels, Height, Width). Matplotlib wants (Height, Width).
    # So we reshape it from (1, 28, 28) to (28, 28)
    img_2d = img_array.reshape(28, 28)
    
    plt.imshow(img_2d, cmap='gray')
    plt.title(f"Target: {dummy_labels[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.show()