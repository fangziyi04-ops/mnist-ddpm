
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 128
epochs = 5
learning_rate = 0.001

torch.manual_seed(42)

os.makedirs("checkpoints", exist_ok=True)

print("Device:", device)


# ==========================================
# 2. CNN classifier
# ==========================================

class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
# ==========================================
    # 3. Load dataset
    # ==========================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )


    # ==========================================
    # 4. Model and optimizer
    # ==========================================

    model = MNISTClassifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ==========================================
    # 5. Training
    # ==========================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = F.cross_entropy(
                logits, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        # Evaluation on real MNIST test images
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                predictions = logits.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f} "
            f"- Test Accuracy: {accuracy:.2f}%"
        )


    # ==========================================
    # 6. Save classifier
    # ==========================================

    torch.save(
        model.state_dict(),
        "checkpoints/mnist_classifier.pth"
    )

    print("Classifier training completed!")
