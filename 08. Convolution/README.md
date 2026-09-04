# GPU Convolution

A practical CUDA project demonstrating 2D convolution using GPU parallel processing.

## What This Project Does

The program creates a 1024 × 1024 grayscale image and applies a 3 × 3 convolution kernel to its pixels using CUDA.

The convolution operation combines each pixel with its neighboring pixels according to the selected kernel.

## Operation

```text
Output Pixel = Sum(Input Neighborhood × Kernel)
