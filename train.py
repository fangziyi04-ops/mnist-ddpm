
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimpleUNet


# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
batch_size = 64
epochs = 10
learning_rate = 0.0002

torch.manual_seed(42)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Training device:", device)


# ==========================================
# 2. Load MNIST
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Training images:", len(dataset))


# ==========================================
# 3. Diffusion schedule
# ==========================================

betas = torch.linspace(
    0.0001, 0.02, T, device=device
)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ==========================================
# 4. Forward diffusion
# ==========================================

def forward_diffusion(x0, t, noise):

    alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)

    xt = (
        torch.sqrt(alpha_bar) * x0
        + torch.sqrt(1 - alpha_bar) * noise
    )

    return xt


# ==========================================
# 5. Model and optimizer
# ==========================================

model = SimpleUNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

loss_history = []


# ==========================================
# 6. Training loop
# ==========================================

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{epochs}"
    )

    for images, labels in progress_bar:

        images = images.to(
            device, non_blocking=True
        )

        # Random timestep for every image
        t = torch.randint(
            0, T, (images.size(0),),
            device=device
        )

        # True Gaussian noise
        noise = torch.randn_like(images)

        # Create noisy images
        noisy_images = forward_diffusion(
            images, t, noise
        )

        # Predict noise
        predicted_noise = model(
            noisy_images, t
        )

        # Calculate MSE loss
        loss = F.mse_loss(
            predicted_noise, noise
        )

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / len(dataloader)
    loss_history.append(average_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Average Loss: {average_loss:.6f}"
    )

    # Save checkpoint after each epoch
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history
        },
        "checkpoints/ddpm_latest.pth"
    )


# ==========================================
# 7. Save final model
# ==========================================

torch.save(
    model.state_dict(),
    "checkpoints/ddpm_final.pth"
)


# ==========================================
# 8. Plot training loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("DDPM Training Loss")
plt.grid(True)

plt.savefig(
    "results/training_loss.png",
    dpi=200
)

plt.show()

print("Training completed!")
