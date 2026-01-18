# Model Initialization Documentation

## Xavier Uniform (Glorot Uniform)

In `sam3/model/model_misc.py`, the `TransformerWrapper` (and other components) use `nn.init.xavier_uniform_` to initialize weights.

### 1. Principle

**Distribution**: **Uniform Distribution** $\mathcal{U}(-a, a)$.

**Formula**:
Weights are sampled from $[-a, a]$, where the bound $a$ is calculated to preserve variance across layers:

$$ a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}} $$

*   **fan_in**: Number of input units (e.g. 1000).
*   **fan_out**: Number of output units (e.g. 1000).
*   **gain**: Scaling factor (default 1.0 for Xavier).

### 2. Visualization

You can visualize this distribution by running the script `scripts/viz_xavier.py`.

It generates a histogram showing that weights are uniformly distributed within the calculated bounds (forming a rectangular shape), which is distinct from the bell curve of Normal initialization.

To generate the plot:
```bash
python scripts/viz_xavier.py
```

This will save the visualization to `docs/sam3/assets/xavier_uniform.png`.