# Image Histogram

A CUDA program that calculates the histogram of a grayscale image using GPU parallel processing.

## What This Project Does

The program:

- Creates a grayscale image
- Divides pixel values into 256 bins
- Calculates the histogram on the GPU
- Uses shared memory and atomic operations
- Verifies the result

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
nvcc ImageHistogram.cu -o ImageHistogram
```

## Run

### Windows

```powershell
.\ImageHistogram.exe
```

### Linux

```bash
./ImageHistogram
```

## Expected Output

```text
Image Histogram
Image Size: 1024 x 1024
Total Pixels: 1048576
Histogram Count: 1048576
Result Verification: PASSED
```

## Project Structure

```text
14. Image Histogram/
├── ImageHistogram.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
