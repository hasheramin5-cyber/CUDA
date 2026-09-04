# GPU Matrix Addition

A practical CUDA project demonstrating element-wise matrix addition using GPU parallel processing.

## What This Project Does

The program creates two 1024 × 1024 matrices and adds corresponding elements using a CUDA kernel.

The operation performed is:

```text
C[row][col] = A[row][col] + B[row][col]
