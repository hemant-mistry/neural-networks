import torch
import matplotlib.pyplot as plt
from number_generator import Generator

# 1. Build the empty Generator factory
generator = Generator()

# 2. Load the trained "brain" (weights) from your hard drive
try:
    generator.load_state_dict(torch.load("alpha_gen_v2.pth", weights_only=True))
    print("Trained brain loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'alpha_gen.pth'.")
    exit()

generator.eval()

print("\n=========================================")

while True:
    user_input = input("\nEnter an alphabet (or 'q' to quit): ")
    
    if user_input.lower() == 'q':
        print("Goodbye!")
        break
        
    try:
        target_alphabet = user_input.upper()
        if len(target_alphabet) != 1 or not target_alphabet.isalpha():
            print("Please enter a single valid letter (A-Z).")
            continue
        
        target_alphabet_ascii = ord(target_alphabet)-65
        noise = torch.randn(3, 100)
        
        labels = torch.full((3,), target_alphabet_ascii, dtype=torch.long)
    
        with torch.no_grad():
            fake_imgs = generator(noise, labels)
            
        fig, axes = plt.subplots(1, 3, figsize=(6, 6))
        fig.suptitle(f"AI Generated: {target_alphabet}", fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flatten()):
            img_2d = fake_imgs[i].view(28, 28).numpy()
            
            ax.imshow(img_2d, cmap='gray')
            ax.axis('off')
            
        plt.tight_layout()
        plt.show() 
        
    except ValueError:
        print("Invalid input. Please type a number.")