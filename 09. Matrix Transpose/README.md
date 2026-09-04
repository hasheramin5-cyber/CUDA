# GPU Matrix Transpose

A practical CUDA project demonstrating matrix transposition using GPU parallel processing and shared memory.

## What This Project Does

The program creates a 1024 × 1024 matrix and transposes it using a CUDA kernel.

Matrix transposition converts rows into columns and columns into rows.

The operation performed is:

```text
Output[col][row] = Input[row][col]
