# GPU Matrix Multiplication

A practical CUDA project demonstrating matrix multiplication using GPU parallel processing.

## What This Project Does

The program multiplies two 512 × 512 matrices using a CUDA kernel.

The operation performed is:

```text
C[row][col] = Σ(A[row][k] × B[k][col])
