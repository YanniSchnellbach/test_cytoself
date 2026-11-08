import torch

# Select GPU if available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

# Create tensors on GPU
x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

# Matrix multiplication
z = torch.matmul(x, y)

print("Result device:", z.device)
print("Result shape:", z.shape)