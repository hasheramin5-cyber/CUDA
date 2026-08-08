<div align="center">

<img src="Assets/cuda-logo.svg" alt="CUDA Logo" width="180"/>

# CUDA

### Practical GPU Computing & CUDA Projects

A collection of practical, ready-to-run CUDA projects focused on
GPU computing, parallel programming, performance acceleration,
and real-world computational workloads.

</div>

---

## Projects

| # | Project | Focus |
| --- | --- | --- |
| 01 | [Vector Addition](./01.%20Vector%20Addition/) | Parallel vector computation |
| 02 | [Vector Multiplication](./02.%20Vector%20Multiplication/) | Element-wise GPU operations |
| 03 | [Matrix Addition](./03.%20Matrix%20Addition/) | 2D GPU computation |
| 04 | [Matrix Multiplication](./04.%20Matrix%20Multiplication/) | Parallel matrix computation |
| 05 | [Parallel Reduction](./05.%20Parallel%20Reduction/) | Parallel aggregation |
| 06 | [Image Blur](./06.%20Image%20Blur/) | GPU image processing |
| 07 | [Edge Detection](./07.%20Edge%20Detection/) | Computer vision acceleration |
| 08 | [Convolution](./08.%20Convolution/) | GPU convolution |
| 09 | [Matrix Transpose](./09.%20Matrix%20Transpose/) | GPU memory optimization |
| 10 | [CPU vs GPU Benchmark](./10.%20CPU%20vs%20GPU%20Benchmark/) | Performance comparison |

---

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- `nvcc` compiler
- C++ compiler

Check your CUDA installation:

```bash
nvcc --version
```

Check your NVIDIA GPU:

```bash
nvidia-smi
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/hasheramin5-cyber/CUDA.git
cd CUDA
```

Navigate to any project:

```bash
cd 01_Vector_Addition
```

Compile:

```bash
nvcc vector_addition.cu -o vector_addition
```

Run on Windows:

```bash
vector_addition.exe
```

Run on Linux:

```bash
./vector_addition
```

Each project contains its own `README.md` with project-specific
requirements, compilation instructions, usage, and expected output.

---

## Focus Areas

- CUDA Programming
- GPU Computing
- Parallel Computing
- GPU Acceleration
- CUDA C/C++
- Computer Vision
- Performance Optimization
- High-Performance Computing

---

## Repository Structure

```text
CUDA/
│
├── Assets/
│   └── cuda-logo.svg
│
├── 01. Vector Addition/
├── 02. Vector Multiplication/
├── 03. Matrix Addition/
├── 04. Matrix Multiplication/
├── 05. Parallel Reduction/
├── 06. Image Blur/
├── 07. Edge Detection/
├── 08. Convolution/
├── 09. Matrix Transpose/
├── 10. CPU vs GPU Benchmark/
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## About

This repository focuses on practical CUDA implementations designed
to demonstrate how NVIDIA GPUs can accelerate computationally
intensive workloads through parallel execution.

The projects are kept independent, reproducible, and focused on
real GPU-computing use cases.

---

## License

This repository is licensed under the [MIT License](./LICENSE).
