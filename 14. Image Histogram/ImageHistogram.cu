#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define WIDTH 1024
#define HEIGHT 1024
#define NUM_BINS 256
#define THREADS_PER_BLOCK 256

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

__global__ void imageHistogramKernel(
    const unsigned char *image,
    unsigned int *histogram,
    int totalPixels)
{
  __shared__ unsigned int localHistogram[NUM_BINS];

  int threadIndex = threadIdx.x;

  if (threadIndex < NUM_BINS)
  {
    localHistogram[threadIndex] = 0;
  }

  __syncthreads();

  int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  int stride =
      blockDim.x * gridDim.x;

  while (index < totalPixels)
  {
    atomicAdd(&localHistogram[image[index]], 1);
    index += stride;
  }

  __syncthreads();

  if (threadIndex < NUM_BINS)
  {
    atomicAdd(
        &histogram[threadIndex],
        localHistogram[threadIndex]);
  }
}

int main()
{
  const int totalPixels = WIDTH * HEIGHT;

  std::vector<unsigned char> hostImage(totalPixels);
  std::vector<unsigned int> hostHistogram(NUM_BINS, 0);

  std::srand(42);

  for (int i = 0; i < totalPixels; ++i)
  {
    hostImage[i] =
        static_cast<unsigned char>(std::rand() % NUM_BINS);
  }

  unsigned char *deviceImage = nullptr;
  unsigned int *deviceHistogram = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceImage,
              totalPixels * sizeof(unsigned char)),
          "cudaMalloc(deviceImage)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceHistogram,
              NUM_BINS * sizeof(unsigned int)),
          "cudaMalloc(deviceHistogram)"))
  {
    cudaFree(deviceImage);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceImage,
              hostImage.data(),
              totalPixels * sizeof(unsigned char),
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceImage);
    cudaFree(deviceHistogram);
    return 1;
  }

  if (!checkCudaError(
          cudaMemset(
              deviceHistogram,
              0,
              NUM_BINS * sizeof(unsigned int)),
          "cudaMemset"))
  {
    cudaFree(deviceImage);
    cudaFree(deviceHistogram);
    return 1;
  }

  int blocks =
      (totalPixels + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  imageHistogramKernel<<<blocks, THREADS_PER_BLOCK>>>(
      deviceImage,
      deviceHistogram,
      totalPixels);

  if (!checkCudaError(
          cudaGetLastError(),
          "imageHistogramKernel launch"))
  {
    cudaFree(deviceImage);
    cudaFree(deviceHistogram);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceImage);
    cudaFree(deviceHistogram);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostHistogram.data(),
              deviceHistogram,
              NUM_BINS * sizeof(unsigned int),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaFree(deviceImage);
    cudaFree(deviceHistogram);
    return 1;
  }

  unsigned long long totalCount = 0;

  for (int bin = 0; bin < NUM_BINS; ++bin)
  {
    totalCount += hostHistogram[bin];
  }

  bool correct =
      (totalCount == static_cast<unsigned long long>(totalPixels));

  std::cout << "Image Histogram\n";
  std::cout << "Image Size: "
            << WIDTH << " x " << HEIGHT << '\n';
  std::cout << "Total Pixels: "
            << totalPixels << '\n';
  std::cout << "Histogram Count: "
            << totalCount << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  if (correct)
  {
    std::cout << "\nHistogram:\n";

    for (int bin = 0; bin < NUM_BINS; ++bin)
    {
      std::cout << bin << ": "
                << hostHistogram[bin] << '\n';
    }
  }

  cudaFree(deviceImage);
  cudaFree(deviceHistogram);

  return correct ? 0 : 1;
}