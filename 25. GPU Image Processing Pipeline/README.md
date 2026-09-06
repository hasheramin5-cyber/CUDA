# GPU Image Processing Pipeline

A CUDA program that processes a grayscale image through multiple GPU image-processing stages.

## What This Project Does

The pipeline:

- Generates grayscale image data
- Applies a blur filter on the GPU
- Applies Sobel edge detection
- Copies the processed image back to the CPU
- Reports basic processing results

## CUDA Concepts

- CUDA kernels
- 2D thread grids
- GPU image processing
- Blur filtering
- Sobel edge detection
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
nvcc GPUImageProcessingPipeline.cu -o GPUImageProcessingPipeline
```

## Run

### Windows

```powershell
.\GPUImageProcessingPipeline.exe
```

### Linux

```bash
./GPUImageProcessingPipeline
```

## Expected Output

```text
GPU Image Processing Pipeline
Image Size: 1024 x 1024
Pipeline: Blur -> Edge Detection
Non-Zero Edge Pixels: XXXXX
Maximum Edge Value: XXX
Processing: PASSED
```

## Project Structure

```text
25. GPU Image Processing Pipeline/
├── GPUImageProcessingPipeline.cu
└── README.md
```

## Important

A CUDA-capable NVIDIA GPU is required to run this project.
