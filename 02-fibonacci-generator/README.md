# Fibonacci Decoder Transformer

![alt text](attn.png)

## Overview

This project trains a decoder-only transformer to predict the next Fibonacci number given a sequence of preceding values.

The model achieves **100% accuracy** on Fibonacci sequence prediction.

If you view the attention heatmap, it is not 100% looking back at n-1, n-2 positions, the model may have found a valid but redundant solution to construct the following token.

## Architecture

Transformer Decoder

- **Embedding**
- **Positional Encoding**
- **Decoder Blocks (4 Layers)**
  - **Multi-Head Attention (2 Heads)**
  - **Feed Forward Network**
  - **Layer Norm + Residual Connections**
- **Linear Out**

## Example training inputs

```
x =   1,   4,   7,   8,  12,  17,  26,  40,  63, 100
      |    |    |    |    |    |    |    |    |    |
      v    v    v    v    v    v    v    v    v    v
y =   4,   7,   8,  12,  17,  26,  40,  63, 100,   2

At position i, looks at x[:i] to predict output y
```

During training, the accuracies of first three positions are ignored `[<SOS>, SEED_1, SEED_2]` since those are given while generating the sequence.

## Example

Seed: `[1, 4]`
Output: `[1, 4, 5, 9, 14, 23, 37, 60, 97]`
