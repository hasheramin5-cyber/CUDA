#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define KERNEL_SIZE 3
#define THREADS_X 16
#define THREADS_Y 16

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void convolution(
    const float *input,
    float *output,
    int width,
    int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
  {
    return;
  }

  const int radius = KERNEL_SIZE / 2;

  float sum = 0.0f;

  for (int kernelY = 0; kernelY < KERNEL_SIZE; ++kernelY)
  {
    for (int kernelX = 0; kernelX < KERNEL_SIZE; ++kernelX)
    {
      int inputX = x + kernelX - radius;
      int inputY = y + kernelY - radius;

      if (inputX >= 0 && inputX < width &&
          inputY >= 0 && inputY < height)
      {
        int inputIndex = inputY * width + inputX;
        int kernelIndex = kernelY * KERNEL_SIZE + kernelX;

        sum += input[inputIndex] * d_kernel[kernelIndex];
      }
    }
  }

  output[y * width + x] = sum;
}

bool checkCudaError(cudaError_t error, const char *operation)
{
  if (error != cudaSuccess)
  {
    std::cerr << "CUDA Error during " << operation << ": "
              << cudaGetErrorString(error) << '\n';
    return false;
  }

  return true;
}

int main()
{
  const int width = IMAGE_WIDTH;
  const int height = IMAGE_HEIGHT;
  const int pixels = width * height;

  const size_t bytes = pixels * sizeof(float);
  const size_t kernelBytes =
      KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

  std::vector<float> h_input(pixels);
  std::vector<float> h_output(pixels);

  const float h_kernel[KERNEL_SIZE * KERNEL_SIZE] =
      {
          1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
          1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
          1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f};

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      h_input[y * width + x] =
          static_cast<float>((x + y) % 256);
    }
  }

  if (!checkCudaError(
          cudaMemcpyToSymbol(
              d_kernel,
              h_kernel,
              kernelBytes),
          "cudaMemcpyToSymbol(kernel)"))
  {
    return 1;
  }

  float *d_input = nullptr;
  float *d_output = nullptr;

  if (!checkCudaError(
          cudaMalloc(&d_input, bytes),
          "cudaMalloc(d_input)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&d_output, bytes),
          "cudaMalloc(d_output)"))
  {
    cudaFree(d_input);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              d_input,
              h_input.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy(input)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  dim3 threadsPerBlock(THREADS_X, THREADS_Y);

  dim3 blocksPerGrid(
      (width + THREADS_X - 1) / THREADS_X,
      (height + THREADS_Y - 1) / THREADS_Y);

  convolution<<<blocksPerGrid, threadsPerBlock>>>(
      d_input,
      d_output,
      width,
      height);

  if (!checkCudaError(
          cudaGetLastError(),
          "kernel launch"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "kernel execution"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              h_output.data(),
              d_output,
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy(output)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  bool valid = true;

  for (float value : h_output)
  {
    if (value < 0.0f || value > 255.0f)
    {
      valid = false;
      break;
    }
  }

  if (valid)
  {
    std::cout << "Convolution completed successfully.\n";
    std::cout << "Image size: "
              << width << " x " << height << '\n';
    std::cout << "Kernel size: "
              << KERNEL_SIZE << " x "
              << KERNEL_SIZE << '\n';

    std::cout << "Sample output pixels:\n";

    for (int i = 0; i < 5; ++i)
    {
      std::cout << "Pixel[" << i << "] = "
                << h_output[i] << '\n';
    }
  }
  else
  {
    std::cerr << "Output verification failed.\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);

  return valid ? 0 : 1;
}