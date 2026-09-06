# Parallel Histogram

A CUDA program that calculates a histogram of values in parallel using GPU threads.

## What This Project Does

The program:

- Generates input data
- Divides values into 256 bins
- Calculates the histogram on the GPU
- Uses shared memory for local histograms
- Verifies the GPU result

## CUDA Concepts

- CUDA kernels
- Shared memory
- Atomic operations
- Parallel processing
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
nvcc ParallelHistogram.cu -o ParallelHistogram
```

## Run

### Windows

```powershell
.\ParallelHistogram.exe
```

### Linux

```bash
./ParallelHistogram
```

## Expected Output

```text
Parallel Histogram
Data Size: 1000000
Number of Bins: 256
Total Count: 1000000
Result Verification: PASSED
```

## Project Structure

```text
13. Parallel Histogram/
├── ParallelHistogram.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
