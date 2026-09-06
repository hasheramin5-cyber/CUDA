# GPU Random Numbers

A CUDA program that generates random numbers in parallel using the cuRAND library.

## What This Project Does

The program:

- Creates GPU random number states
- Generates random floating-point values
- Stores the values in GPU memory
- Copies the results to the CPU
- Calculates basic statistics

## CUDA Concepts

- cuRAND
- Random number generation
- CUDA kernels
- Parallel processing
- Device memory
- CUDA Runtime API

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- cuRAND
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc GPURandomNumbers.cu -o GPURandomNumbers
```

## Run

### Windows

```powershell
.\GPURandomNumbers.exe
```

### Linux

```bash
./GPURandomNumbers
```

## Expected Output

```text
GPU Random Numbers
Number of Values: 1000000
First 10 Values:
0.XXXXXX
0.XXXXXX
0.XXXXXX
...
Minimum: 0.XXXXXX
Maximum: 1.XXXXXX
Average: 0.XXXXXX
```

## Project Structure

```text
18. GPU Random Numbers/
├── GPURandomNumbers.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
