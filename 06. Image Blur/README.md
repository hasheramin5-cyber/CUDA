# GPU Image Blur

A practical CUDA project demonstrating image blur using GPU parallel processing.

## What This Project Does

The program creates a 1024 × 1024 RGB image and applies a box blur to its pixels using a CUDA kernel.

Each GPU thread processes one pixel and calculates the average value of neighboring pixels.

## Operation

```text
Output Pixel = Average of Neighboring Pixels
