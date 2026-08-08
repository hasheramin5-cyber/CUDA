# GPU Vector Addition

A practical CUDA project demonstrating how vector addition can be performed in parallel using an NVIDIA GPU.

## What This Project Does

The program creates two vectors containing **1,000,000 elements** and calculates their element-wise sum using a CUDA kernel.

The operation performed is:

```text
C[i] = A[i] + B[i]
```

Instead of processing every element sequentially on the CPU, CUDA assigns different elements to different GPU threads.

## How It Works

```text
Host (CPU)
    │
    ├── Create Vector A
    ├── Create Vector B
    │
    ▼
Device (GPU)
    │
    ├── Allocate GPU memory
    ├── Copy A and B to GPU
    │
    ├── Launch CUDA kernel
    │      │
    │      ├── Thread 0 → C[0]
    │      ├── Thread 1 → C[1]
    │      ├── Thread 2 → C[2]
    │      └── ...
    │
    └── Copy C back to CPU
           │
           ▼
        Verify result
```

## CUDA Concepts Used

* CUDA kernels
* GPU threads
* Thread indexing
* Blocks and grids
* Host memory
* Device memory
* `cudaMalloc`
* `cudaMemcpy`
* `cudaFree`
* Kernel launching
* CUDA error handling
* Result verification

## Requirements

You need:

* NVIDIA GPU with CUDA support
* NVIDIA GPU driver
* CUDA Toolkit
* `nvcc` CUDA compiler
* C++ compiler

Check your CUDA compiler:

```bash
nvcc --version
```

Check your NVIDIA GPU:

```bash
nvidia-smi
```

## Compilation

Open a terminal inside this project directory and run:

```bash
nvcc vector_addition.cu -o vector_addition
```

## Run

### Windows

```bash
vector_addition.exe
```

### Linux

```bash
./vector_addition
```

## Expected Output

A successful execution should produce output similar to:

```text
Vector addition completed successfully.
Elements processed: 1000000
Sample results:
C[0] = 0
C[1] = 3
C[2] = 6
C[3] = 9
C[4] = 12
```

The exact output may vary depending on the environment, but the verification should report successful completion.

## Why Use CUDA?

Vector addition is highly parallel because every element can be calculated independently.

For example:

```text
A[0] + B[0] → C[0]
A[1] + B[1] → C[1]
A[2] + B[2] → C[2]
A[3] + B[3] → C[3]
```

These operations can be distributed across many GPU threads.

This makes vector addition a simple example of how CUDA can transform an independent workload into a parallel GPU computation.

## Thread Indexing

Each CUDA thread calculates its global index using:

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
```

Where:

* `blockIdx.x` identifies the current block
* `blockDim.x` represents the number of threads per block
* `threadIdx.x` identifies the current thread inside the block

The program uses:

```cpp
#define THREADS_PER_BLOCK 256
```

The number of blocks is calculated dynamically:

```cpp
int blocksPerGrid =
    (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
```

This ensures that enough threads are launched to process the complete vector.

## Error Handling

CUDA API calls and kernel execution are checked for errors.

The program checks:

* Memory allocation
* Memory transfers
* Kernel launch
* Kernel execution
* Result verification

This prevents CUDA failures from being silently ignored.

## Important Note

This project requires an **NVIDIA CUDA-capable GPU** for execution.

The source code is designed for a CUDA-enabled environment and is not expected to execute with GPU acceleration on systems without an NVIDIA CUDA GPU.
