#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define DATA_SIZE 1000000
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

__global__ void histogramKernel(
    const unsigned char *data,
    unsigned int *histogram,
    size_t size)
{
  __shared__ unsigned int localHistogram[NUM_BINS];

  int threadIndex = threadIdx.x;

  if (threadIndex < NUM_BINS)
  {
    localHistogram[threadIndex] = 0;
  }

  __syncthreads();

  size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  size_t stride =
      static_cast<size_t>(blockDim.x) * gridDim.x;

  while (index < size)
  {
    atomicAdd(&localHistogram[data[index]], 1);
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
  std::vector<unsigned char> hostData(DATA_SIZE);
  std::vector<unsigned int> hostHistogram(NUM_BINS, 0);

  std::srand(42);

  for (size_t i = 0; i < DATA_SIZE; ++i)
  {
    hostData[i] =
        static_cast<unsigned char>(std::rand() % NUM_BINS);
  }

  unsigned char *deviceData = nullptr;
  unsigned int *deviceHistogram = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceData,
              DATA_SIZE * sizeof(unsigned char)),
          "cudaMalloc(deviceData)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceHistogram,
              NUM_BINS * sizeof(unsigned int)),
          "cudaMalloc(deviceHistogram)"))
  {
    cudaFree(deviceData);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceData,
              hostData.data(),
              DATA_SIZE * sizeof(unsigned char),
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceData);
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
    cudaFree(deviceData);
    cudaFree(deviceHistogram);
    return 1;
  }

  int blocks =
      (DATA_SIZE + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  histogramKernel<<<blocks, THREADS_PER_BLOCK>>>(
      deviceData,
      deviceHistogram,
      DATA_SIZE);

  if (!checkCudaError(
          cudaGetLastError(),
          "histogramKernel launch"))
  {
    cudaFree(deviceData);
    cudaFree(deviceHistogram);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceData);
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
    cudaFree(deviceData);
    cudaFree(deviceHistogram);
    return 1;
  }

  bool correct = true;

  for (int bin = 0; bin < NUM_BINS; ++bin)
  {
    unsigned int expected = 0;

    for (size_t i = 0; i < DATA_SIZE; ++i)
    {
      if (hostData[i] == bin)
      {
        ++expected;
      }
    }

    if (hostHistogram[bin] != expected)
    {
      correct = false;
      break;
    }
  }

  unsigned long long totalCount = 0;

  for (int bin = 0; bin < NUM_BINS; ++bin)
  {
    totalCount += hostHistogram[bin];
  }

  std::cout << "Parallel Histogram\n";
  std::cout << "Data Size: "
            << DATA_SIZE << '\n';
  std::cout << "Number of Bins: "
            << NUM_BINS << '\n';
  std::cout << "Total Count: "
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

  cudaFree(deviceData);
  cudaFree(deviceHistogram);

  return correct ? 0 : 1;
}