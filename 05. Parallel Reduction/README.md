# Parallel Reduction

A practical CUDA project demonstrating parallel reduction for summing a large vector using GPU threads and shared memory.

## What This Project Does

The program creates a vector containing 1,048,576 floating-point values and calculates their total sum using a CUDA reduction kernel.

The reduction combines many values into a smaller number of partial results until a final sum is produced.

## CUDA Concepts

- CUDA kernels
- Thread indexing
- Blocks and grids
- Shared memory
- Thread synchronization
- `__syncthreads()`
- Parallel reduction
- Device memory
- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`
- CUDA error handling
- Result verification

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- `nvcc` compiler
- C++ compiler

Check the CUDA compiler:

```bash
nvcc --version
