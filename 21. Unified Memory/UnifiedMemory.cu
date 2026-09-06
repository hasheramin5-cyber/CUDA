#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define NUM_ELEMENTS 1048576
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

__global__ void processKernel(
    const float *input,
    float *output,
    int size)
{
  int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    output[index] =
        input[index] * 2.0f + 1.0f;
  }
}

int main()
{
  const size_t bytes =
      NUM_ELEMENTS * sizeof(float);

  float *managedInput = nullptr;
  float *managedOutput = nullptr;

  if (!checkCudaError(
          cudaMallocManaged(
              &managedInput,
              bytes),
          "cudaMallocManaged(input)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMallocManaged(
              &managedOutput,
              bytes),
          "cudaMallocManaged(output)"))
  {
    cudaFree(managedInput);
    return 1;
  }

  for (int i = 0; i < NUM_ELEMENTS; ++i)
  {
    managedInput[i] =
        static_cast<float>(i);
  }

  int blocks =
      (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  processKernel<<<blocks, THREADS_PER_BLOCK>>>(
      managedInput,
      managedOutput,
      NUM_ELEMENTS);

  if (!checkCudaError(
          cudaGetLastError(),
          "processKernel launch"))
  {
    cudaFree(managedInput);
    cudaFree(managedOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(managedInput);
    cudaFree(managedOutput);
    return 1;
  }

  bool correct = true;

  for (int i = 0; i < NUM_ELEMENTS; ++i)
  {
    float expected =
        managedInput[i] * 2.0f + 1.0f;

    if (std::fabs(managedOutput[i] - expected) > 0.001f)
    {
      correct = false;
      break;
    }
  }

  std::cout << "Unified Memory\n";
  std::cout << "Number of Elements: "
            << NUM_ELEMENTS << '\n';
  std::cout << "Memory Allocation: "
            << "cudaMallocManaged"
            << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaFree(managedInput);
  cudaFree(managedOutput);

  return correct ? 0 : 1;
}