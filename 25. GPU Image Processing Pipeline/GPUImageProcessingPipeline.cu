#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define WIDTH 1024
#define HEIGHT 1024
#define THREADS_PER_BLOCK 16

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

__global__ void blurKernel(
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

  int sum = 0;
  int count = 0;

  for (int dy = -1; dy <= 1; ++dy)
  {
    for (int dx = -1; dx <= 1; ++dx)
    {
      int nx = x + dx;
      int ny = y + dy;

      if (nx >= 0 && nx < width &&
          ny >= 0 && ny < height)
      {
        sum += input[ny * width + nx];
        ++count;
      }
    }
  }

  output[y * width + x] =
      static_cast<unsigned char>(sum / count);
}

__global__ void edgeDetectionKernel(
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

  if (x == 0 || y == 0 ||
      x == width - 1 || y == height - 1)
  {
    output[y * width + x] = 0;
    return;
  }

  int gx =
      -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

  int gy =
      -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

  int magnitude =
      abs(gx) + abs(gy);

  magnitude =
      min(magnitude, 255);

  output[y * width + x] =
      static_cast<unsigned char>(magnitude);
}

int main()
{
  const size_t imageSize =
      WIDTH * HEIGHT * sizeof(unsigned char);

  std::vector<unsigned char> hostInput(
      WIDTH * HEIGHT);

  std::vector<unsigned char> hostOutput(
      WIDTH * HEIGHT);

  for (int y = 0; y < HEIGHT; ++y)
  {
    for (int x = 0; x < WIDTH; ++x)
    {
      hostInput[y * WIDTH + x] =
          static_cast<unsigned char>(
              (x + y) % 256);
    }
  }

  unsigned char *deviceInput = nullptr;
  unsigned char *deviceBlurred = nullptr;
  unsigned char *deviceOutput = nullptr;

  if (!checkCudaError(
          cudaMalloc(&deviceInput, imageSize),
          "cudaMalloc(deviceInput)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceBlurred, imageSize),
          "cudaMalloc(deviceBlurred)"))
  {
    cudaFree(deviceInput);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceOutput, imageSize),
          "cudaMalloc(deviceOutput)"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceInput,
              hostInput.data(),
              imageSize,
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    cudaFree(deviceOutput);
    return 1;
  }

  dim3 blockSize(
      THREADS_PER_BLOCK,
      THREADS_PER_BLOCK);

  dim3 gridSize(
      (WIDTH + blockSize.x - 1) / blockSize.x,
      (HEIGHT + blockSize.y - 1) / blockSize.y);

  blurKernel<<<gridSize, blockSize>>>(
      deviceInput,
      deviceBlurred,
      WIDTH,
      HEIGHT);

  if (!checkCudaError(
          cudaGetLastError(),
          "blurKernel launch"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    cudaFree(deviceOutput);
    return 1;
  }

  edgeDetectionKernel<<<gridSize, blockSize>>>(
      deviceBlurred,
      deviceOutput,
      WIDTH,
      HEIGHT);

  if (!checkCudaError(
          cudaGetLastError(),
          "edgeDetectionKernel launch"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    cudaFree(deviceOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    cudaFree(deviceOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostOutput.data(),
              deviceOutput,
              imageSize,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceBlurred);
    cudaFree(deviceOutput);
    return 1;
  }

  unsigned int nonZeroPixels = 0;
  unsigned char maximumValue = 0;

  for (unsigned char value : hostOutput)
  {
    if (value > 0)
    {
      ++nonZeroPixels;
    }

    maximumValue =
        std::max(maximumValue, value);
  }

  std::cout << "GPU Image Processing Pipeline\n";
  std::cout << "Image Size: "
            << WIDTH << " x "
            << HEIGHT << '\n';

  std::cout << "Pipeline: "
            << "Blur -> Edge Detection\n";

  std::cout << "Non-Zero Edge Pixels: "
            << nonZeroPixels << '\n';

  std::cout << "Maximum Edge Value: "
            << static_cast<int>(maximumValue)
            << '\n';

  std::cout << "Processing: PASSED\n";

  cudaFree(deviceInput);
  cudaFree(deviceBlurred);
  cudaFree(deviceOutput);

  return 0;
}