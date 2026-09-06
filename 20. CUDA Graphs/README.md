# CUDA Graphs

A CUDA program that captures a GPU kernel operation in a CUDA Graph and executes the graph.

## What This Project Does

The program:

- Allocates GPU memory
- Creates a CUDA Graph
- Adds a kernel node
- Instantiates the graph
- Launches the graph
- Verifies the result

## CUDA Concepts

- CUDA Graphs
- Graph nodes
- Graph instantiation
- Graph execution
- CUDA streams
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
nvcc CUDAGraphs.cu -o CUDAGraphs
```

## Run

### Windows

```powershell
.\CUDAGraphs.exe
```

### Linux

```bash
./CUDAGraphs
```

## Expected Output

```text
CUDA Graphs
Number of Elements: 1048576
Graph Execution: PASSED
```

## Project Structure

```text
20. CUDA Graphs/
├── CUDAGraphs.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
