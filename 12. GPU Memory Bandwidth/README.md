# GPU Memory Bandwidth

A CUDA program that measures the memory bandwidth of an NVIDIA GPU.

## What This Project Does

The program:

- Allocates GPU memory
- Copies data to GPU memory
- Runs a CUDA copy kernel
- Measures execution time
- Calculates memory bandwidth
- Verifies the result

## CUDA Concepts

- Global memory
- CUDA kernel
- CUDA events
- Memory bandwidth
- Device memory
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
nvcc GPUMemoryBandwidth.cu -o GPUMemoryBandwidth
```

## Run

### Windows

```powershell
.\GPUMemoryBandwidth.exe
```

### Linux

```bash
./GPUMemoryBandwidth
```

## Expected Output

```text
GPU Memory Bandwidth
Data Size: X.XX GB
Kernel Time: X.XX ms
Estimated Bandwidth: XXX.XX GB/s
Result Verification: PASSED
```

Actual bandwidth depends on the GPU.

## Project Structure

```text
12. GPU Memory Bandwidth/
├── GPUMemoryBandwidth.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
