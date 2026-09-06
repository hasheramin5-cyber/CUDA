# Monte Carlo Simulation

A CUDA program that uses the Monte Carlo method to estimate the value of Pi using GPU parallel processing.

## What This Project Does

The program:

- Generates random points on the GPU
- Checks whether points are inside a circle
- Counts the points inside the circle
- Estimates the value of Pi

## CUDA Concepts

- CUDA kernels
- Parallel processing
- cuRAND
- Random number generation
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
nvcc MonteCarloSimulation.cu -o MonteCarloSimulation
```

## Run

### Windows

```powershell
.\MonteCarloSimulation.exe
```

### Linux

```bash
./MonteCarloSimulation
```

## Expected Output

```text
Monte Carlo Simulation
Number of Samples: 10000000
Points Inside Circle: XXXXXXX
Estimated Value of Pi: 3.14XXXX
```

The estimated value may be slightly different on each run.

## Project Structure

```text
15. Monte Carlo Simulation/
├── MonteCarloSimulation.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
