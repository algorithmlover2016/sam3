import numpy as np
import matplotlib.pyplot as plt
import os

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    # Approximation of GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def glu_silu(x):
    # GLU in the scalar case where gate = input, essentially SiLU/Swish
    # GLU(x, g) = x * sigmoid(g). If we view the activation function part, it is often x * sigmoid(x) (Swish/SiLU)
    return x * (1 / (1 + np.exp(-x)))

def plot_activations():
    x = np.linspace(-4, 4, 1000)
    
    y_relu = relu(x)
    y_gelu = gelu(x)
    y_glu = glu_silu(x)

    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y_relu, label='ReLU', linestyle='--', alpha=0.7, linewidth=2)
    plt.plot(x, y_gelu, label='GELU', linewidth=2)
    plt.plot(x, y_glu, label='GLU (SiLU)', linestyle='-.', alpha=0.8, linewidth=2)

    plt.title('Activation Functions Comparison')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)

    # Ensure directory exists
    # Assuming script is run from project root, or we handle relative paths
    # Target: docs/sam3/assets/activation_functions.svg
    
    # Let's try to resolve absolute path to be safe, or relative to this script
    # Script is in docs/scripts/
    # Target is in docs/sam3/assets/
    # So relative path from script dir is: ../sam3/assets
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level (docs) then into sam3/assets
    output_dir = os.path.join(script_dir, "../sam3/assets")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "activation_functions.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    plot_activations()