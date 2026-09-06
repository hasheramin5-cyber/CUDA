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

  float *hostInput = nullptr;
  float *hostOutput = nullptr;

  if (!checkCudaError(
          cudaMallocHost(
              &hostInput,
              bytes),
          "cudaMallocHost(input)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMallocHost(
              &hostOutput,
              bytes),
          "cudaMallocHost(output)"))
  {
    cudaFreeHost(hostInput);
    return 1;
  }

  for (int i = 0; i < NUM_ELEMENTS; ++i)
  {
    hostInput[i] =
        static_cast<float>(i);
  }

  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceInput,
              bytes),
          "cudaMalloc(deviceInput)"))
  {
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceOutput,
              bytes),
          "cudaMalloc(deviceOutput)"))
  {
    cudaFree(deviceInput);
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceInput,
              hostInput,
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  int blocks =
      (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  processKernel<<<blocks, THREADS_PER_BLOCK>>>(
      deviceInput,
      deviceOutput,
      NUM_ELEMENTS);

  if (!checkCudaError(
          cudaGetLastError(),
          "processKernel launch"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostOutput,
              deviceOutput,
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFreeHost(hostInput);
    cudaFreeHost(hostOutput);
    return 1;
  }

  bool correct = true;

  for (int i = 0; i < NUM_ELEMENTS; ++i)
  {
    float expected =
        hostInput[i] * 2.0f + 1.0f;

    if (std::fabs(hostOutput[i] - expected) > 0.001f)
    {
      correct = false;
      break;
    }
  }

  std::cout << "Pinned Memory\n";
  std::cout << "Number of Elements: "
            << NUM_ELEMENTS << '\n';
  std::cout << "Host Memory: "
            << "cudaMallocHost"
            << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFreeHost(hostInput);
  cudaFreeHost(hostOutput);

  return correct ? 0 : 1;
}