# TransformerDecoderLayer Documentation

This document details the architecture and data flow of the `TransformerDecoderLayer` class defined in `sam3/model/decoder.py`. 

This implementation separates the attention mechanisms into distinct blocks (Self-Attention, Text Cross-Attention, and Image Cross-Attention) and includes explicit support for the **DAC (Divide-and-Conquer)** decoding strategy and a **Presence Token**.

## Overview

The `TransformerDecoderLayer` processes query features through a sequence of attention mechanisms and a feed-forward network.

### Key Features
1.  **DAC Strategy Support**: Explicitly handles splitting queries into "One-to-One" (O2O) and "One-to-Many" (O2M) groups for self-attention.
2.  **Multi-Modal Support**: Includes an optional **Text Cross-Attention** block to condition queries on text embeddings.
3.  **Presence Token**: Handles a special token used to predict the presence of objects.
4.  **Pre-Norm/Post-Norm**: This specific implementation generally follows a **Post-Norm**-like structure (Add -> Norm), but with specific placements for `norm1`, `norm2`, `norm3`.

## Initialization Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `d_model` | `int` | Dimension of the embeddings (e.g., 256). |
| `dim_feedforward` | `int` | Inner dimension of the FFN. |
| `dropout` | `float` | Dropout probability. |
| `activation` | `str` | Activation function name (e.g., 'relu', 'gelu'). |
| `n_heads` | `int` | Number of attention heads. |
| `self_attention` | `nn.MultiheadAttention` | (Implicitly created) Standard MHA. |
| `cross_attention` | `nn.Module` | External module for image cross-attention. |
| `use_text_cross_attention` | `bool` | Whether to enable text cross-attention. |

## Data Flow & Shape Analysis

**Notation:**
- $N_q$: Number of Queries (`tgt` length)
- $B$: Batch Size
- $D$: Model Dimension (`d_model`)
- $L$: Image Memory Sequence Length ($H \times W$)
- $L_{txt}$: Text Memory Sequence Length

### 1. Self-Attention Block (with DAC)

This block allows queries to attend to each other.

*   **DAC Split**:
    If `dac=True`, `tgt` is split into:
    *   `tgt_o2o` (First half, $N_q/2$): Participates in self-attention.
    *   `tgt_o2m` (Second half, $N_q/2$): Bypasses self-attention.
    
    If `presence_token` is provided, it is prepended to `tgt_o2o` (and stripped later).

*   **Operation**:
    $$ Q = K = \text{with\_pos\_embed}(tgt\_o2o, query\_pos\_o2o) $$
    $$ tgt2 = \text{SelfAttn}(Q, K, V=tgt\_o2o) $$
    $$ tgt\_o2o = tgt\_o2o + \text{Dropout2}(tgt2) $$
    $$ tgt\_active = \text{Norm2}(tgt\_o2o) $$

    > **Note on Shape Requirements**:
    > The positional embedding addition (`tgt + query_pos`) requires specific shape alignment:
    > *   **Sequence Length (Dim 0)**: Must match exactly between `tgt` and `query_pos` because they are sliced simultaneously during DAC processing.
    > *   **Feature Dimension (Dim -1)**: Must match exactly.
    > *   **Batch Dimension (Dim 1)**: Can differ, as PyTorch broadcasting allows `(N, 1, D)` position encodings to be added to `(N, B, D)` targets.

*   **Recombine**:
    If `dac=True`, `tgt` is reconstructed by concatenating `tgt_active` and `tgt_o2m`.

### 2. Text Cross-Attention (Optional)

If `use_text_cross_attention=True`:

*   **Inputs**: `tgt` ($N_q, B, D$), `memory_text` ($L_{txt}, B, D$).
*   **Operation**:
    $$ Q = tgt + query\_pos $$
    $$ K = V = memory\_text $$
    $$ tgt2 = \text{MultiheadAttn}(Q, K, V) $$
    $$ tgt = tgt + \text{Dropout}(tgt2) $$
    $$ tgt = \text{Norm}(tgt) $$

    > **Matrix Calculation Insight**:
    > Ideally, with Batch ($B$) processed first:
    > 1.  Inputs: $Q$ is $(B, N_q, D)$, $K$ is $(B, L_{txt}, D)$.
    > 2.  **Dot Product**: Compute affinity between visual queries and text tokens.
    >     $$ (B, N_q, D) \times (B, L_{txt}, D)^T \rightarrow (B, N_q, L_{txt}) $$
    > 3.  **Result**: The resulting $(B, N_q, L_{txt})$ matrix represents the attention scores (how much each query "cares" about each word).

### 3. Image Cross-Attention

*   **Inputs**: `tgt` ($N_q, B, D$), `memory` ($L, B, D$).
*   **Operation**:
    $$ Q = tgt + query\_pos $$
    $$ K = memory + pos $$
    $$ V = memory $$
    $$ tgt2 = \text{CrossAttn}(Q, K, V) $$
    $$ tgt = tgt + \text{Dropout1}(tgt2) $$
    $$ tgt = \text{Norm1}(tgt) $$

### 4. Feed-Forward Network (FFN)

*   **Operation**:
    $$ tgt2 = \text{Linear2}(\text{Dropout3}(\text{Activation}(\text{Linear1}(tgt)))) $$
    $$ tgt = tgt + \text{Dropout4}(tgt2) $$
    $$ tgt = \text{Norm3}(tgt) $$

### 5. FFN Implementation Details

The feed-forward block utilizes specific design choices for numerical stability and performance:

1.  **Disabled Autocast**:
    To prevent precision issues (overflow/NaN) during the dimension expansion (`Linear1`) and non-linear activation, the block is forced to run in full precision (`float32`) by disabling AMP (`enabled=False`).

2.  **No Activation on Output**:
    The sequence is `Linear1 -> Activation -> Linear2`. The final projection (`Linear2`) does **not** have an activation function. This allows the residual update ($\Delta x$) to span the full range of values (positive and negative), which is crucial for the additive residual connection ($x + \Delta x$).

3.  **Dual Dropout**:
    *   **Activation Dropout (`dropout3`)**: Applied *inside* the hidden layer (after activation). This regularizes the internal feature representation, preventing co-adaptation of hidden units.
    *   **Residual Dropout (`dropout4`)**: Applied to the *output* of the FFN before collecting it into the residual stream. This forces the network to be robust to the loss of entire features in the update step.

### 6. Presence Tokens & Queries Explained

To clarify the roles of different inputs:

*   **Queries ($N_q$)**:
    *   These are **Architecurally Fixed Vectors** (Hyperparameters, e.g., 4), typically learned during training.
    *   They do **not** correspond to user inputs or pixels.
    *   Instead, they act as "Question Slots" or "Hypotheses" (e.g., "Is there a small object?", "Is there a large object?", "Is there a part here?"). This allows the model to handle ambiguity by outputting multiple potential masks for a single prompt.

*   **Batch Size ($B$)**:
    *   This corresponds to the number of **User Prompts / Independent Tasks**.
    *   Formula: $B = N_{Images} \times N_{Prompts\_per\_Image}$.
    *   Each batch element tracks one unique target object.

*   **Memory / Feature Map**:
    *   **Per-Prompt Assignment**: Each Prompt (Batch Element $b$) gets its own copy of the Image Feature Map (Memory). If multiple prompts share an image, the image features are expanded/repeated to match $B$.
    *   **Shared by Queries**: Inside one batch element $b$, all $N_q$ queries share and attend to the **same** Memory map. They work collaboratively on the same image features to resolve the same prompt.

*   **Presence Token**:
    *   **Shape**: `(1, B, D)`.
    *   **Role**: There is **one** presence token per Batch Element (Target Object). It acts as a "Team Leader" for the $N_q$ queries, aggregating information to answer a global question: "Is the object targeted by Prompt $b$ actually present/visible in the current frame?"
    *   **Logic (Master Switch)**:
        *   It determines the **Global Existence** of the instance.
        *   **Present**: If the object is even partially visible (e.g., only the head of a person is visible), it should predict "Present". The specific queries will then handle segmentation (e.g., Query 1 segments the head).
        *   **Absent**: If the object is fully occluded or out of frame, it predicts "Absent". This acts as a master gate to suppress all mask outputs for this instance, regardless of what the individual queries might predict.

## Visual Workflow

```mermaid
graph TD
    subgraph Inputs
        Tgt(Target / Queries<br/>N_q, B, D)
        QPos(Query Pos)
        Mem(Image Memory<br/>L, B, D)
        MPos(Image Pos)
        TxtMem(Text Memory)
        PresTok(Presence Token)
    end

    IsDAC{DAC Enabled?}
    HasPres{Has Presence Token?}

    %% DAC Split Logic
    Split["Split Queries<br/>O2O: First N/2<br/>O2M: Last N/2"]
    
    %% Self Attention Handling
    subgraph SA_Prep [Prepare Self-Attention]
        SliceQPos[Slice Query Pos]
        PrependPres[Prepend Presence Token]
    end

    subgraph SelfAttentionBlock ["Self Attention Block"]
        direction TB
        AddPosSA[Add Pos Embed]
        SA[Self Attention]
        DropSA[Dropout2]
        ResSA[Residual Add]
        NormSA[Norm2]
        
        AddPosSA -- Q, K --> SA
        SA --> DropSA --> ResSA --> NormSA
    end

    Recombine[Recombine: O2O + O2M]

    %% Text Cross Attention
    subgraph TextAttnBlock ["Text Cross-Attention (Optional)"]
        direction TB
        AddPosTx[Add Q-Pos]
        TxAttn[Text Attention]
        DropTx[Dropout]
        ResTx[Residual Add]
        NormTx[Norm]
        
        AddPosTx -- Q --> TxAttn
        TxAttn --> DropTx --> ResTx --> NormTx
    end

    %% Image Cross Attention
    subgraph ImgAttnBlock ["Image Cross-Attention"]
        direction TB
        AddPosImgQ[Add Q-Pos]
        AddPosImgK[Add K-Pos]
        ImgAttn[Cross Attention]
        DropImg[Dropout1]
        ResImg[Residual Add]
        NormImg[Norm1]

        AddPosImgQ -- Q --> ImgAttn
        AddPosImgK -- K --> ImgAttn
        ImgAttn --> DropImg --> ResImg --> NormImg
    end

    %% FFN
    subgraph FFNBlock ["Feed Forward Network"]
        direction TB
        Lin1[Linear1]
        Act[Activation]
        Lin2[Linear2]
        DropFFN[Dropout3/4]
        ResFFN[Residual Add]
        NormFFN[Norm3]
        
        Lin1 --> Act --> Lin2 --> DropFFN --> ResFFN --> NormFFN
    end

    %% Connections
    Tgt --> IsDAC
    IsDAC -- Yes --> Split
    IsDAC -- No --> HasPres
    
    Split -- O2O --> SliceQPos --> PrependPres
    QPos --> SliceQPos

    PrependPres --> AddPosSA
    PresTok --> PrependPres

    %% SA Bypass
    Split -- O2M --> Recombine
    NormSA --> Recombine

    %% Post SA
    Recombine --> TextAttnBlock
    HasPres -- No --> AddPosSA
    
    TextAttnBlock --> ImgAttnBlock
    ImgAttnBlock --> FFNBlock
    
    %% Cross Attn Connections
    QPos -.-> AddPosTx
    TxtMem -.-> TxAttn

    QPos -.-> AddPosImgQ
    Mem --> AddPosImgK
    MPos -.-> AddPosImgK
    Mem -.-> ImgAttn

    %% Output
    FFNBlock --> ExtractPres{Extract Presence?}
    ExtractPres -- Yes --> OutPres(Presence Output)
    ExtractPres -- Yes (Rest) --> OutTgt(Queries Output)