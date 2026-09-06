# FFT Processing

A CUDA program that performs a Fast Fourier Transform using the cuFFT library.

## What This Project Does

The program:

- Generates a sample signal
- Transfers the signal to the GPU
- Performs a forward FFT
- Finds the strongest frequency component
- Displays the frequency bin and magnitude

## CUDA Concepts

- cuFFT
- Fast Fourier Transform
- Complex numbers
- GPU signal processing
- Device memory
- CUDA Runtime API

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA GPU Driver
- CUDA Toolkit
- cuFFT
- `nvcc`

```bash
nvcc --version
```

```bash
nvidia-smi
```

## Compile

```bash
nvcc FFTProcessing.cu -o FFTProcessing -lcufft
```

## Run

### Windows

```powershell
.\FFTProcessing.exe
```

### Linux

```bash
./FFTProcessing
```

## Expected Output

```text
FFT Processing
Signal Size: 4096
Strongest Frequency Bin: XX
Magnitude: XXXXX.XX
```

## Project Structure

```text
17. FFT Processing/
├── FFTProcessing.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
