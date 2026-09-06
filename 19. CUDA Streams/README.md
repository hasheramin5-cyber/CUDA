# CUDA Streams

A CUDA program that uses multiple CUDA streams to process data asynchronously.

## What This Project Does

The program:

- Divides data into multiple chunks
- Creates four CUDA streams
- Performs asynchronous memory transfers
- Processes each chunk using a CUDA kernel
- Verifies the final result

## CUDA Concepts

- CUDA streams
- Asynchronous memory transfers
- Kernel execution
- Concurrent operations
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
nvcc CUDAStreams.cu -o CUDAStreams
```

## Run

### Windows

```powershell
.\CUDAStreams.exe
```

### Linux

```bash
./CUDAStreams
```

## Expected Output

```text
CUDA Streams
Number of Elements: 1048576
Number of Streams: 4
Result Verification: PASSED
```

## Project Structure

```text
19. CUDA Streams/
├── CUDAStreams.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
