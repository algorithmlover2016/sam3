# TransformerDecoder Documentation

This document provides a detailed analysis of the `TransformerDecoder` class defined in `sam3/model/decoder.py`. This class orchestrates the multi-layer decoding process, managing iterative box refinement, coordinate-based auxiliary encodings (Sine/RoPE), and the flow of queries through the network.

## 1. Overview

The `TransformerDecoder` essentially wraps a stack of `TransformerDecoderLayer` instances. However, it is not just a container; it actively manages:
1.  **Iterative Box Refinement**: Updates query reference points (bounding boxes) after every layer.
2.  **Coordinate Encodings**: dynamic generation of `query_pos` based on current reference boxes.
3.  **Presence Tokens**: Manages global instance existence flags.
4.  **Box Relative Position Bias (BoxRPB)**: dynamic generation of attention masks based on box geometry.

## 2. Parameter Analysis

Here we estimate the parameter count. Assume $D$=`d_model` (e.g., 256).

### 2.1 Component Parameters

| Component | Symbol | Shape / Definition | Approx Params (D=256) |
| :--- | :--- | :--- | :--- |
| **Layers** | $L \times \Theta_{layer}$ | $L$ copies of `TransformerDecoderLayer` | $L \times 3.15M$ |
| **Query Embed** | $W_{query}$ | Embedding $(N_q, D)$ | $4 \times 256 \approx 1K$ |
| **Box Head** | $W_{box}$ | MLP: $D \to D \to D \to 4$ (3 layers) | $D^2 + D^2 + 4D \approx 132K$ |
| **Ref Point Head** | $W_{ref}$ | MLP: $2D \to D \to D \to D(pos)$ | $2D^2 + D^2 + D^2 \approx 262K$ |
| **Ref Points Embed**| $W_{ref\_pts}$ | Embedding $(N_q, 4)$ | $4 \times 4 = 16$ |
| **Pres Token** | $W_{pres}$ | Embedding $(1, D)$ + MLP Head | $\approx 66K$ |
| **BoxRPB** | $W_{rpb}$ | 2 MLPs: $2 \to D \to H_{attn}$ | Small ($\approx 10K$) |

### 2.2 Total Parameter Count Estimate
Assuming $L=6$ standard layers:
*   **Layers**: ~19M params.
*   **Heads & Embeds**: ~0.5M params.
*   **Total**: $\approx 19.5M$ parameters.

## 3. Data Flow & Shapes

Let:
*   $N_q$ = Number of Queries (e.g., 4).
*   $B$ = Batch Size.
*   $D$ = `d_model` (256).
*   $L_{mem}$ = Feature Map Size ($H \times W$, e.g., 4096).
*   $N_{layer}$ = Number of decoder layers (e.g., 6).

### 3.1 Initialization (Before Layer Loop)

1.  **Outputs Initialization** (`output`):
    *   Input `tgt`: $(N_q, B, D)$.
    *   If `tgt` is from `query_embed`: $(4, B, 256)$.

2.  **Reference Boxes Initialization** (`reference_boxes`):
    *   If provided: Use input.
    *   If None (One-Stage): Init from `self.reference_points`.
    *   Shape: $(N_q, B, 4)$. (Sigmoid applied).

### 3.2 Inside the Layer Loop (Per Layer $i$)

For each layer $i \in [0, N_{layer}-1]$:

1.  **Coordinate Embedding Generation**:
    *   Input: `reference_boxes` $(N_q, B, 4)$.
    *   Sine Positional Encoding: Maps coords to $(N_q, B, 2D)$.
    *   `ref_point_head`: Projects back to Query Pos $(N_q, B, D)$.
    *   **Result**: `query_pos` $(N_q, B, D)$.

2.  **Box RPB (Optional)**:
    *   Calculates relative offsets between all pixels in Memory ($H, W$) and Box Centers.
    *   Generates `memory_mask` (Bias).
    *   Shape: $(B \cdot Heads, N_q, L_{mem})$.

3.  **Layer Processing**:
    *   Call `layer(tgt=output, query_pos=query_pos, memory=memory, ...)`.
    *   **Input Shapes**: `output` $(N_q, B, D)$, `memory` $(L_{mem}, B, D)$.
    *   **Output Shape**: `output` $(N_q, B, D)$ (Updated features).

4.  **Box Refinement (Iterative)**:
    *   **Box Head**: `delta = box_head(output)`.
    *   **Update**: `new_boxes = sigmoid(inverse_sigmoid(old_boxes) + delta)`.
    *   **Shape**: `new_boxes` $(N_q, B, 4)$.
    *   *The `reference_boxes` for the NEXT layer will be `new_boxes`.*

5.  **Presence Prediction (Optional)**:
    *   Input: `presence_out` from layer (Token 0).
    *   Head: `presence_logits` $(1, B, 1)$.

### 3.3 Final Output

The model returns stacks of intermediate results from all layers:
1.  `intermediate_outputs`: $(N_{layer}, N_q, B, D)$.
2.  `intermediate_ref_boxes`: $(N_{layer}, N_q, B, 4)$.

## 4. Forward Workflow (Mermaid)

```mermaid
graph TD
    subgraph Inputs
        Tgt(tgt<br/>Nq, B, D)
        Mem(memory<br/>L, B, D)
        RefInit(reference_points)
    end

    subgraph InitBlock [Initialization]
        GetRef[Init Reference Boxes<br/>Use RefInit or Input]
        AssignOut[output = tgt]
    end

    subgraph LayerLoop [Loop: Layers 0 to N-1]
        GenPos["Gen Query Pos<br/>SineEmbed + MLP<br/>Uses current Ref Box"]
        GenRPB["Gen Box RPB Mask<br/>(Optional)<br/>(B*H, Nq, L)"]
        
        LayerFwd["TransformerDecoderLayer<br/>Self Attn + Cross Attn + FFN"]
        
        BoxHead["Box Head MLP<br/>D -> 4"]
        BoxUpd["Box Refinement<br/>Sigmoid(Unsig(Ref) + Delta)"]
        
        Store[Store Intermediates]
    end

    subgraph OutputBlock [Outputs]
        Stack[Stack Intermediates]
        Return((Return<br/>Hidden States<br/>Boxes))
    end

    Tgt --> AssignOut
    RefInit -.-> GetRef
    GetRef --> GenPos

    AssignOut --> LayerFwd
    Mem --> LayerFwd
    
    GenPos -- query_pos (Nq, B, D) --> LayerFwd
    GenRPB -- memory_mask --> LayerFwd
    getRef -- reference_boxes --> GenPos
    getRef -- reference_boxes --> GenRPB

    LayerFwd -- output (Nq, B, D) --> BoxHead
    BoxHead -- delta --> BoxUpd
    GetRef -- old_boxes --> BoxUpd
    BoxUpd -- new_boxes --> Store
    
    BoxUpd -. Next Iteration RefBoxes .-> GenPos
    LayerFwd -. Next Iteration Output .-> LayerFwd

    Store --> Stack --> Return