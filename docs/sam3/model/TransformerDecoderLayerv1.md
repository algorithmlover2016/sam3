# TransformerDecoderLayerv1 Documentation

This document details the architecture and data flow of the `TransformerDecoderLayerv1` class defined in `sam3/model/decoder.py`. This class implements a standard Transformer Decoder layer with support for Pre-Norm/Post-Norm variants and a specific Divide-and-Conquer (DAC) decoding strategy.

## Overview

The `TransformerDecoderLayerv1` is a composite layer consisting of three main sub-blocks:
1.  **Self-Attention**: Processes interactions between queries.
2.  **Cross-Attention**: Processes interactions between queries (features) and memory (image features).
3.  **Feed-Forward Network (FFN)**: Applies a point-wise **MLP (Multilayer Perceptron)** to the features.

> **Note on MLP**: In the context of Transformers, "MLP" typically refers to a **Position-wise Feed-Forward Network**. It consists of two linear transformations with a non-linear activation function in between. It processes each token position independently and identically, distinguishing it from attention layers which mix information across the sequence.
>
> **Why is it called MLP?**
> The term "Multilayer Perceptron" (MLP) historically refers to a neural network with at least one hidden layer (and thus multiple layers of neurons/perceptrons). Although the block here is simple (Input $\to$ Hidden $\to$ Output), it technically satisfies the definition of an MLP. In modern Transformer literature, "FFN" (Feed-Forward Network) and "MLP" are used interchangeably to describe this block.

It supports conditional positional encoding injection and a split-processing strategy for queries known as "DAC" (Divide-and-Conquer).

## Initialization Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `d_model` | `int` | The dimension of the input/output embeddings (e.g., 256). |
| `dim_feedforward` | `int` | The inner dimension of the FFN (e.g., 2048). |
| `dropout` | `float` | Dropout probability. |
| `activation` | `str` | Activation function used in the FFN (e.g., 'relu'). |
| `pre_norm` | `bool` | If `True`, applies LayerNorm before attention/MLP blocks (Pre-Norm). If `False`, applies after (Post-Norm). |
| `self_attention` | `nn.Module` | The module used for self-attention. |
| `cross_attention` | `nn.Module` | The module used for cross-attention. |
| `pos_enc_at_attn` | `bool` | Whether to add positional encoding to Query/Key in Self-Attention. |
| `pos_enc_at_cross_attn_queries` | `bool` | Whether to add positional encoding to Query in Cross-Attention. |
| `pos_enc_at_cross_attn_keys` | `bool` | Whether to add positional encoding to Key in Cross-Attention. |

## Data Flow & Shape Analysis

**Notation:**
- $B$: Batch Size
- $N_q$: Number of Queries (`tgt` length)
- $L$: Memory Sequence Length ($Memory Spatial Height \times Width$)
- $D$: Model Dimension (`d_model`)
- $D_{ffn}$: Feedforward Dimension (`dim_feedforward`)

### 1. Inputs

The `forward_pre` method (Pre-Norm variant) accepts the following primary tensors:

- **Target (`tgt`)**: Shape $(N_q, B, D)$ representing the query features.
- **Memory (`memory`)**: Shape $(L, B, D)$ representing the image context features.
- **Query Pos (`query_pos`)**: Shape $(N_q, B, D)$ representing positional encodings for the queries.
- **Memory Pos (`pos`)**: Shape $(L, B, D)$ representing positional encodings for the memory.

### 2. Forward Logic (Pre-Norm Variant)

The layer sequentially processes the input through Self-Attention, Cross-Attention, and FFN.

#### Phase 1: Self-Attention
This block allows queries to attend to each other.

1.  **DAC Split (Optional)**:
    If `dac=True`, the input `tgt` is split into two halves along the query dimension ($N_q$). Self-attention is applied **only to the first half** ($N_q/2$).
    - `active_tgt`: $0$ to $N_q/2$
    - `passive_tgt`: $N_q/2$ to $N_q$ (bypasses self-attention)

2.  **Normalization**:
    $$ X_{norm} = \text{LayerNorm1}(tgt) $$

3.  **Positional Encoding Injection**:
    - Query ($Q$): $X_{norm} + query\_pos$ (if `pos_enc_at_attn` is True, else $X_{norm}$)
    - Key ($K$): $X_{norm} + query\_pos$ (if `pos_enc_at_attn` is True, else $X_{norm}$)
    - Value ($V$): $X_{norm}$

    > **Note on Q/K Identity**: The variables `q` and `k` passed here are the **Input Embeddings** (original token features + position encoding). They are *not* yet the projected Query/Key vectors used for dot-product attention.
    > The `MultiheadAttention` module internally performs the projection:
    > *   $\text{Internal } Q = \text{Input } q \times W_q$
    > *   $\text{Internal } K = \text{Input } k \times W_k$
    >
    > Therefore, even though the **Input Embeddings** for `q` and `k` are the same tensor (`tgt + pos`), the **Internal Projected Vectors** are distinct due to independent weights ($W_q \neq W_k$).

4.  **Attention**:
    $$ X_{attn} = \text{SelfAttention}(Q, K, V) $$

5.  **Residual & Dropout**:
    $$ tgt = tgt + \text{Dropout1}(X_{attn}) $$

6.  **DAC Recombine (Optional)**:
    If `dac=True`, the processed `active_tgt` is concatenated back with the unprocessed `passive_tgt` to restore the full $(N_q, B, D)$ shape.

#### Phase 2: Cross-Attention
This block allows queries to extract information from the memory (image features).

1.  **Normalization**:
    $$ X_{norm} = \text{LayerNorm2}(tgt) $$

2.  **Positional Encoding Injection**:
    - Query ($Q$): $X_{norm} + query\_pos$ (if `pos_enc_at_cross_attn_queries` is True, else $X_{norm}$)
    - Key ($K$): $memory + pos$ (if `pos_enc_at_cross_attn_keys` is True, else $memory$)
    - Value ($V$): $memory$

3.  **Attention**:
    $$ X_{attn} = \text{CrossAttention}(Q, K, V) $$

4.  **Residual & Dropout**:
    $$ tgt = tgt + \text{Dropout2}(X_{attn}) $$

#### Phase 3: Feed-Forward Network (FFN)
Standard MLP block processing each token independently.

1.  **Normalization**:
    $$ X_{norm} = \text{LayerNorm3}(tgt) $$

2.  **MLP Operation**:
    $$ X_{mlp} = \text{Linear2}(\text{Dropout}(\text{Activation}(\text{Linear1}(X_{norm})))) $$
    - Linear1: Projects $D \to D_{ffn}$.
    - Linear2: Projects $D_{ffn} \to D$.

3.  **Residual & Dropout**:
    $$ tgt = tgt + \text{Dropout3}(X_{mlp}) $$

## Visual Workflow (Pre-Norm Variant with DAC)

This specific workflow illustrates the **Pre-Norm** architecture where `LayerNorm` is applied *before* the sub-layer operations (Attention/MLP). It also depicts the optional **DAC (Divide-and-Conquer)** strategy where self-attention is only applied to the first half of the queries.

```mermaid
graph TD
    subgraph Inputs
        Tgt(Target / Queries<br/>N_q, B, D)
        Mem(Memory / Image<br/>L, B, D)
        QPos(Query Pos)
        MPos(Memory Pos)
    end

    IsDAC{DAC Enabled?}

    %% DAC Split Logic
    Split["Split Queries<br/>Active: First N/2<br/>Passive: Last N/2"]
    
    %% Self Attention Block (Pre-Norm)
    subgraph SelfAttentionBlock ["Self Attention Block (Pre-Norm)"]
        direction TB
        input_sa((Input))
        LN1[LayerNorm 1]
        AddPosSA[Add Q/K-Pos]
        SA[Self Attention]
        Drop1[Dropout]
        Res1[Residual Add]
        
        input_sa --> Res1
        input_sa --> LN1 --> AddPosSA
        
        AddPosSA -- Q, K --> SA
        LN1 -- V --> SA
        
        SA --> Drop1 --> Res1
    end
    
    PassivePath(Passive Branch<br/>Identity)

    Recombine[Recombine / Concatenate]

    %% Cross Attention Block (Pre-Norm)
    subgraph CrossAttentionBlock ["Cross Attention Block (Pre-Norm)"]
        direction TB
        input_ca((Input))
        LN2[LayerNorm 2]
        AddPosCA_Q[Add Q-Pos]
        AddPosCA_K[Add K-Pos]
        CA[Cross Attention]
        Drop2[Dropout]
        Res2[Residual Add]
        
        input_ca --> Res2
        input_ca --> LN2 --> AddPosCA_Q -- Q --> CA
        
        Mem --> AddPosCA_K -- K --> CA
        Mem -- V --> CA
        MPos -.-> AddPosCA_K
        
        CA --> Drop2 --> Res2
    end

    %% FFN Block (Pre-Norm)
    subgraph FFNBlock ["Feed Forward Block (Pre-Norm)"]
        direction TB
        input_ffn((Input))
        LN3[LayerNorm 3]
        Lin1[Linear 1]
        Act[Activation]
        Lin2[Linear 2]
        Drop3[Dropout]
        Res3[Residual Add]
        
        input_ffn --> Res3
        input_ffn --> LN3 --> Lin1 --> Act --> Drop3 --> Lin2 --> Res3
    end

    %% Connections
    Tgt --> IsDAC
    QPos -.-> AddPosSA
    QPos -.-> AddPosCA_Q

    %% Logic Flow
    IsDAC -- No --> input_sa
    IsDAC -- Yes --> Split
    
    Split -- Active Tgt --> input_sa
    Split -- Passive Tgt --> Recombine

    Res1 --> Recombine
    
    %% If DAC=No, Res1 goes straight to next stage.
    %% Graphically we can merge the paths:
    %% If DAC=No, Recombine is effectively Identity/Pass-through of the full tensor.
    
    Res1 -.-> IsDAC_Check{Path}
    IsDAC_Check -- No DAC --> input_ca
    Recombine -- With DAC --> input_ca

    Res2 --> input_ffn
    Res3 --> Output(Output<br/>N_q, B, D)
```

## Visual Workflow (Post-Norm Variant)

This workflow illustrates the **Post-Norm** architecture where `LayerNorm` is applied *after* the residual addition of the sub-layer operations. Note that the **DAC** strategy is **not supported/used** in the Post-Norm implementation of this layer.

```mermaid
graph TD
    subgraph Inputs
        Tgt(Target / Queries<br/>N_q, B, D)
        Mem(Memory / Image<br/>L, B, D)
        QPos(Query Pos)
        MPos(Memory Pos)
    end

    %% Self Attention Block (Post-Norm)
    subgraph SelfAttentionBlock ["Self Attention Block (Post-Norm)"]
        direction TB
        input_sa((Input))
        AddPosSA[Add Pos]
        SA[Self Attention]
        Drop1[Dropout]
        Res1[Residual Add]
        LN1[LayerNorm 1]
        
        input_sa --> Res1
        input_sa --> AddPosSA
        
        AddPosSA -- Q, K --> SA
        input_sa -- V --> SA
        
        SA --> Drop1 --> Res1
        Res1 --> LN1
    end

    %% Cross Attention Block (Post-Norm)
    subgraph CrossAttentionBlock ["Cross Attention Block (Post-Norm)"]
        direction TB
        input_ca((Input))
        AddPosCA_Q[Add Q-Pos]
        AddPosCA_K[Add K-Pos]
        CA[Cross Attention]
        Drop2[Dropout]
        Res2[Residual Add]
        LN2[LayerNorm 2]
        
        input_ca --> Res2
        input_ca --> AddPosCA_Q -- Q --> CA
        
        Mem --> AddPosCA_K -- K --> CA
        Mem -- V --> CA
        MPos -.-> AddPosCA_K
        QPos -.-> AddPosCA_Q
        
        CA --> Drop2 --> Res2
        Res2 --> LN2
    end

    %% FFN Block (Post-Norm)
    subgraph FFNBlock ["Feed Forward Block (Post-Norm)"]
        direction TB
        input_ffn((Input))
        Lin1[Linear 1]
        Act[Activation]
        Lin2[Linear 2]
        Drop3[Dropout]
        Res3[Residual Add]
        LN3[LayerNorm 3]
        
        input_ffn --> Res3
        input_ffn --> Lin1 --> Act --> Drop3 --> Lin2 --> Res3
        Res3 --> LN3
    end

    %% Connections
    Tgt --> input_sa
    QPos -.-> AddPosSA

    LN1 --> input_ca
    LN2 --> input_ffn
    LN3 --> Output(Output<br/>N_q, B, D)
```

## Summary of Operations

| Layer | Input Shape | Operation | Output Shape | Parameters (Main) |
| :--- | :--- | :--- | :--- | :--- |
| **Norm 1** | $(N_q, B, D)$ | Layer Normalization | $(N_q, B, D)$ | $2 \times D$ |
| **Self Attn** | $(N_q, B, D)$ | Multi-Head Attention | $(N_q, B, D)$ | $4 \times D^2$ (approx) |
| **Norm 2** | $(N_q, B, D)$ | Layer Normalization | $(N_q, B, D)$ | $2 \times D$ |
| **Cross Attn** | Q:$(N_q, B, D)$, KV:$(L, B, D)$ | Multi-Head Attention | $(N_q, B, D)$ | $4 \times D^2$ (approx) |
| **Norm 3** | $(N_q, B, D)$ | Layer Normalization | $(N_q, B, D)$ | $2 \times D$ |
| **Linear 1** | $(N_q, B, D)$ | Linear Projection | $(N_q, B, D_{ffn})$ | $D \times D_{ffn}$ |
| **Linear 2** | $(N_q, B, D_{ffn})$ | Linear Projection | $(N_q, B, D)$ | $D_{ffn} \times D$ |