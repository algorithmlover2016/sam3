import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Ensure directory exists
os.makedirs("docs/sam3/assets", exist_ok=True)

# 1. Create a large Linear layer
linear = nn.Linear(1000, 1000)

# 2. Apply Xavier Uniform
nn.init.xavier_uniform_(linear.weight)

# 3. Flatten weights
weights = linear.weight.detach().numpy().flatten()

# 4. Plot
plt.figure(figsize=(10, 6))
plt.hist(weights, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Weights Initialized with Xavier Uniform")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.5)

# Calculate theoretical bound a
fan_in, fan_out = 1000, 1000
gain = 1.0
a = gain * (6.0 / (fan_in + fan_out))**0.5
print(f"Theoretical Bound a = +/- {a:.5f}")

# Plot bounds
plt.axvline(x=-a, color='red', linestyle='--', label=f'Bound -a (-{a:.4f})')
plt.axvline(x=a, color='red', linestyle='--', label=f'Bound a ({a:.4f})')
plt.legend()

# Save instead of show
output_path = "docs/sam3/assets/xavier_uniform.png"
plt.savefig(output_path)
print(f"Chart saved to {output_path}")