#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define THREADS_X 16
#define THREADS_Y 16

__global__ void edgeDetection(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
  {
    return;
  }

  int index = y * width + x;

  if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
  {
    output[index] = 0;
    return;
  }

  int topLeft = input[(y - 1) * width + (x - 1)];
  int top = input[(y - 1) * width + x];
  int topRight = input[(y - 1) * width + (x + 1)];

  int left = input[y * width + (x - 1)];
  int right = input[y * width + (x + 1)];

  int bottomLeft = input[(y + 1) * width + (x - 1)];
  int bottom = input[(y + 1) * width + x];
  int bottomRight = input[(y + 1) * width + (x + 1)];

  int gx =
      -topLeft - 2 * left - bottomLeft + topRight + 2 * right + bottomRight;

  int gy =
      -topLeft - 2 * top - topRight + bottomLeft + 2 * bottom + bottomRight;

  float magnitude = sqrtf(
      static_cast<float>(gx * gx + gy * gy));

  if (magnitude > 255.0f)
  {
    magnitude = 255.0f;
  }

  output[index] =
      static_cast<unsigned char>(magnitude);
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
  const size_t bytes = pixels * sizeof(unsigned char);

  std::vector<unsigned char> h_input(pixels);
  std::vector<unsigned char> h_output(pixels);

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      if (x > width / 3 && x < (width * 2) / 3)
      {
        h_input[y * width + x] = 255;
      }
      else
      {
        h_input[y * width + x] = 0;
      }
    }
  }

  unsigned char *d_input = nullptr;
  unsigned char *d_output = nullptr;

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

  edgeDetection<<<blocksPerGrid, threadsPerBlock>>>(
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

  for (unsigned char pixel : h_output)
  {
    if (pixel > 255)
    {
      valid = false;
      break;
    }
  }

  if (valid)
  {
    std::cout << "Edge detection completed successfully.\n";
    std::cout << "Image size: "
              << width << " x " << height << '\n';

    std::cout << "Sample output pixels:\n";

    for (int i = 0; i < 5; ++i)
    {
      std::cout << "Pixel[" << i << "] = "
                << static_cast<int>(h_output[i]) << '\n';
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