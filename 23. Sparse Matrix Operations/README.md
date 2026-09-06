# Sparse Matrix Operations

A CUDA program that performs sparse matrix-vector multiplication using the cuSPARSE library.

## What This Project Does

The program:

- Creates a sparse matrix in CSR format
- Allocates GPU memory
- Transfers sparse matrix data to the GPU
- Uses cuSPARSE for matrix-vector multiplication
- Copies the result back to the CPU
- Verifies the result

## CUDA Concepts

- cuSPARSE
- Sparse matrices
- CSR format
- Sparse matrix-vector multiplication
- Device memory
- CUDA Runtime API

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- cuSPARSE
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc SparseMatrixOperations.cu -o SparseMatrixOperations -lcusparse
```

## Run

### Windows

```powershell
.\SparseMatrixOperations.exe
```

### Linux

```bash
./SparseMatrixOperations
```

## Expected Output

```text
Sparse Matrix Operations
Matrix Size: 4 x 5
Non-Zero Elements: 7
Result Verification: PASSED
```

## Project Structure

```text
23. Sparse Matrix Operations/
├── SparseMatrixOperations.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
