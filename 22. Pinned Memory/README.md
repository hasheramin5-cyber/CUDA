# Pinned Memory

A CUDA program that uses page-locked host memory for CPU and GPU data transfers.

## What This Project Does

The program:

- Allocates pinned host memory
- Allocates GPU memory
- Transfers data to the GPU
- Processes data using a CUDA kernel
- Transfers the result back
- Verifies the result

## CUDA Concepts

- Pinned memory
- `cudaMallocHost`
- Host-to-device transfers
- Device-to-host transfers
- CUDA kernels
- Device memory

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
nvcc PinnedMemory.cu -o PinnedMemory
```

## Run

### Windows

```powershell
.\PinnedMemory.exe
```

### Linux

```bash
./PinnedMemory
```

## Expected Output

```text
Pinned Memory
Number of Elements: 1048576
Host Memory: cudaMallocHost
Result Verification: PASSED
```

## Project Structure

```text
22. Pinned Memory/
├── PinnedMemory.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
