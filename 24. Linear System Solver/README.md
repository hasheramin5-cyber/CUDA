# Linear System Solver

A CUDA program that solves a dense linear system using the cuSOLVER library.

## What This Project Does

The program:

- Creates a 3 × 3 linear system
- Allocates GPU memory
- Performs LU factorization
- Solves the linear system
- Copies the solution back to the CPU
- Verifies the result

## CUDA Concepts

- cuSOLVER
- LU factorization
- Linear system solving
- Device memory
- CUDA Runtime API
- CUDA error handling

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- cuSOLVER
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc LinearSystemSolver.cu -o LinearSystemSolver -lcusolver
```

## Run

### Windows

```powershell
.\LinearSystemSolver.exe
```

### Linux

```bash
./LinearSystemSolver
```

## Expected Output

```text
Linear System Solver
Matrix Size: 3 x 3
Solution: 1 2 3
Result Verification: PASSED
```

## Project Structure

```text
24. Linear System Solver/
├── LinearSystemSolver.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
