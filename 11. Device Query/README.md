# Device Query

A CUDA program that detects NVIDIA GPUs and displays their device information.

## What This Project Does

The program displays:

- GPU name
- Compute capability
- Global memory
- Shared memory
- Warp size
- Thread and grid limits
- Multiprocessors
- Clock rates
- Memory bus width
- CUDA memory capabilities

## CUDA Concepts

- CUDA Runtime API
- Device enumeration
- Device properties
- Compute capability
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
nvcc DeviceQuery.cu -o DeviceQuery
```

## Run

### Windows

```powershell
.\DeviceQuery.exe
```

### Linux

```bash
./DeviceQuery
```

## Expected Output

```text
CUDA Device Query
CUDA-capable devices: 1

Device 0
Name: NVIDIA GPU
Compute Capability: X.X
Total Global Memory: X.XX MB (X.XX GB)
Shared Memory per Block: X.XX MB (X.XX GB)
Registers per Block: XXXX
Warp Size: XX
Maximum Threads per Block: XXX
Multiprocessors: XX
Clock Rate: XXXX MHz
Memory Clock Rate: XXXX MHz
Memory Bus Width: XXX bits
Concurrent Kernels: Yes
Unified Addressing: Yes
Managed Memory: Yes

Current CUDA Device: 0
```

## Project Structure

```text
11. Device Query/
├── DeviceQuery.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
