# cuBLAS Matrix Multiplication

A CUDA program that performs matrix multiplication using the cuBLAS library.

## What This Project Does

The program:

- Creates two matrices
- Allocates GPU memory
- Copies matrices to the GPU
- Uses `cublasSgemm` for matrix multiplication
- Copies the result back to the CPU
- Verifies the result

## CUDA Concepts

- cuBLAS
- `cublasSgemm`
- GPU matrix multiplication
- Device memory
- CUDA Runtime API
- CUDA error handling

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- cuBLAS
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc cuBLASMatrixMultiplication.cu -o cuBLASMatrixMultiplication -lcublas
```

## Run

### Windows

```powershell
.\cuBLASMatrixMultiplication.exe
```

### Linux

```bash
./cuBLASMatrixMultiplication
```

## Expected Output

```text
cuBLAS Matrix Multiplication
Matrix Size: 1024 x 1024
Expected Value: 2048
Result Verification: PASSED
```

## Project Structure

```text
16. cuBLAS Matrix Multiplication/
├── cuBLASMatrixMultiplication.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
