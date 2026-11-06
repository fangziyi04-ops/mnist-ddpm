
import torch
import torchvision
import numpy as np
import matplotlib

# 1. Check installed versions
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# 2. Check computing device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Computing device:", device)

# 3. Create a random tensor
x = torch.randn(3, 3)
print("\nRandom tensor:")
print(x)

# 4. Test a simple tensor operation
y = x * 2
print("\nTensor multiplied by 2:")
print(y)

print("\nEnvironment setup successful!")


# GPU test
print("\n--- GPU Test ---")

print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda")

    # Create tensors on GPU
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    c = torch.matmul(a, b)

    print("Result device:", c.device)
    print("Result shape:", c.shape)
    print("GPU computation successful!")
else:
    print("GPU is not available.")
