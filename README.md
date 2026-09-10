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

| #  | Project                                                                       | Focus                            |
| -- | ----------------------------------------------------------------------------- | -------------------------------- |
| 01 | [Vector Addition](./01.%20Vector%20Addition/)                                 | Parallel vector computation      |
| 02 | [Vector Multiplication](./02.%20Vector%20Multiplication/)                     | Element-wise GPU operations      |
| 03 | [Matrix Addition](./03.%20Matrix%20Addition/)                                 | 2D GPU computation               |
| 04 | [Matrix Multiplication](./04.%20Matrix%20Multiplication/)                     | Parallel matrix computation      |
| 05 | [Parallel Reduction](./05.%20Parallel%20Reduction/)                           | Parallel aggregation             |
| 06 | [Image Blur](./06.%20Image%20Blur/)                                           | GPU image processing             |
| 07 | [Edge Detection](./07.%20Edge%20Detection/)                                   | Computer vision acceleration     |
| 08 | [Convolution](./08.%20Convolution/)                                           | GPU convolution                  |
| 09 | [Matrix Transpose](./09.%20Matrix%20Transpose/)                               | GPU memory optimization          |
| 10 | [CPU vs GPU Benchmark](./10.%20CPU%20vs%20GPU%20Benchmark/)                   | Performance comparison           |
| 11 | [Device Query](./11.%20Device%20Query/)                                       | GPU device information           |
| 12 | [GPU Memory Bandwidth](./12.%20GPU%20Memory%20Bandwidth/)                     | GPU memory performance           |
| 13 | [Parallel Histogram](./13.%20Parallel%20Histogram/)                           | Parallel histogram computation   |
| 14 | [Image Histogram](./14.%20Image%20Histogram/)                                 | GPU image histogram              |
| 15 | [Monte Carlo Simulation](./15.%20Monte%20Carlo%20Simulation/)                 | GPU-based simulation             |
| 16 | [cuBLAS Matrix Multiplication](./16.%20cuBLAS%20Matrix%20Multiplication/)     | GPU linear algebra               |
| 17 | [FFT Processing](./17.%20FFT%20Processing/)                                   | Frequency-domain processing      |
| 18 | [GPU Random Numbers](./18.%20GPU%20Random%20Numbers/)                         | GPU random number generation     |
| 19 | [CUDA Streams](./19.%20CUDA%20Streams/)                                       | Asynchronous GPU execution       |
| 20 | [CUDA Graphs](./20.%20CUDA%20Graphs/)                                         | Graph-based execution            |
| 21 | [Unified Memory](./21.%20Unified%20Memory/)                                   | Unified memory management        |
| 22 | [Pinned Memory](./22.%20Pinned%20Memory/)                                     | Fast host-device transfers       |
| 23 | [Sparse Matrix Operations](./23.%20Sparse%20Matrix%20Operations/)             | Sparse linear algebra            |
| 24 | [Linear System Solver](./24.%20Linear%20System%20Solver/)                     | GPU linear system solving        |
| 25 | [GPU Image Processing Pipeline](./25.%20GPU%20Image%20Processing%20Pipeline/) | Multi-stage GPU image processing |

---

## Requirements

* NVIDIA GPU with CUDA support
* NVIDIA GPU Driver
* CUDA Toolkit
* `nvcc` compiler
* C++ compiler

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

Navigate to a project:

```bash
cd "01. Vector Addition"
```

Compile:

```bash
nvcc VectorAddition.cu -o VectorAddition
```

Run on Windows:

```powershell
.\VectorAddition.exe
```

Run on Linux:

```bash
./VectorAddition
```

Each project contains its own `README.md` with project-specific requirements, compilation instructions, usage, and expected output.

---

## Focus Areas

* CUDA Programming
* GPU Computing
* Parallel Computing
* GPU Acceleration
* CUDA C/C++
* Computer Vision
* Performance Optimization
* High-Performance Computing

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
├── 11. Device Query/
├── 12. GPU Memory Bandwidth/
├── 13. Parallel Histogram/
├── 14. Image Histogram/
├── 15. Monte Carlo Simulation/
├── 16. cuBLAS Matrix Multiplication/
├── 17. FFT Processing/
├── 18. GPU Random Numbers/
├── 19. CUDA Streams/
├── 20. CUDA Graphs/
├── 21. Unified Memory/
├── 22. Pinned Memory/
├── 23. Sparse Matrix Operations/
├── 24. Linear System Solver/
├── 25. GPU Image Processing Pipeline/
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── cuda_error.md
│   │   └── feature_request.md
│   └── pull_request_template.md
│
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── README.md
```

---

## About

This repository focuses on practical CUDA implementations designed to demonstrate how NVIDIA GPUs can accelerate computationally intensive workloads through parallel execution.

The projects are kept independent, reproducible, and focused on real GPU-computing use cases.

---

## License

This repository is licensed under the MIT License.

