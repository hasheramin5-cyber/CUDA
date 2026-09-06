# Unified Memory

A CUDA program that uses Unified Memory to share data between the CPU and GPU.

## What This Project Does

The program:

- Allocates memory using `cudaMallocManaged`
- Initializes data on the CPU
- Processes the data on the GPU
- Accesses the result directly from the CPU
- Verifies the result

## CUDA Concepts

- Unified Memory
- `cudaMallocManaged`
- CUDA kernels
- CPU and GPU memory access
- Device synchronization
- CUDA error handling

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc UnifiedMemory.cu -o UnifiedMemory
```

## Run

### Windows

```powershell
.\UnifiedMemory.exe
```

### Linux

```bash
./UnifiedMemory
```

## Expected Output

```text
Unified Memory
Number of Elements: 1048576
Memory Allocation: cudaMallocManaged
Result Verification: PASSED
```

## Project Structure

```text
21. Unified Memory/
├── UnifiedMemory.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
