# GPU Edge Detection

A practical CUDA project demonstrating edge detection using GPU parallel processing.

## What This Project Does

The program creates a 1024 × 1024 grayscale image and detects edges using the Sobel operator in a CUDA kernel.

The Sobel operator calculates horizontal and vertical intensity changes to identify areas where strong image transitions occur.

## Operation

```text
Gx = Horizontal Intensity Gradient
Gy = Vertical Intensity Gradient

Edge Magnitude = √(Gx² + Gy²)
