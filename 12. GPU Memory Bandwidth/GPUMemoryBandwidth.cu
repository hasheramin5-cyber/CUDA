#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define DATA_SIZE (64 * 1024 * 1024)
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

__global__ void copyKernel(
    const float *input,
    float *output,
    size_t elements)
{
  size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (index < elements)
  {
    output[index] = input[index];
  }
}

int main()
{
  const size_t elements = DATA_SIZE / sizeof(float);
  const size_t bytes = elements * sizeof(float);

  float *hostInput = new float[elements];
  float *hostOutput = new float[elements];

  for (size_t i = 0; i < elements; ++i)
  {
    hostInput[i] = static_cast<float>(i);
  }

  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;

  if (!checkCudaError(
          cudaMalloc(&deviceInput, bytes),
          "cudaMalloc(deviceInput")))
    {
      delete[] hostInput;
      delete[] hostOutput;
      return 1;
    }

  if (!checkCudaError(
          cudaMalloc(&deviceOutput, bytes),
          "cudaMalloc(deviceOutput")))
    {
      cudaFree(deviceInput);
      delete[] hostInput;
      delete[] hostOutput;
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
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  const int blocks =
      static_cast<int>(
          (elements + THREADS_PER_BLOCK - 1) /
          THREADS_PER_BLOCK);

  cudaEvent_t start;
  cudaEvent_t stop;

  if (!checkCudaError(
          cudaEventCreate(&start),
          "cudaEventCreate(start)"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  if (!checkCudaError(
          cudaEventCreate(&stop),
          "cudaEventCreate(stop)"))
  {
    cudaEventDestroy(start);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  if (!checkCudaError(
          cudaEventRecord(start),
          "cudaEventRecord(start)"))
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  copyKernel<<<blocks, THREADS_PER_BLOCK>>>(
      deviceInput,
      deviceOutput,
      elements);

  if (!checkCudaError(
          cudaGetLastError(),
          "copyKernel launch"))
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  if (!checkCudaError(
          cudaEventRecord(stop),
          "cudaEventRecord(stop)"))
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  if (!checkCudaError(
          cudaEventSynchronize(stop),
          "cudaEventSynchronize"))
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  float milliseconds = 0.0f;

  if (!checkCudaError(
          cudaEventElapsedTime(
              &milliseconds,
              start,
              stop),
          "cudaEventElapsedTime"))
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;
    return 1;
  }

  bool correct = true;

  for (size_t i = 0; i < elements; ++i)
  {
    if (hostOutput[i] != hostInput[i])
    {
      correct = false;
      break;
    }
  }

  double seconds = milliseconds / 1000.0;
  double gigabytes =
      static_cast<double>(bytes) /
      (1024.0 * 1024.0 * 1024.0);

  double bandwidth =
      (2.0 * gigabytes) / seconds;

  std::cout << "GPU Memory Bandwidth\n";
  std::cout << "Data Size: "
            << std::fixed << std::setprecision(2)
            << gigabytes << " GB\n";
  std::cout << "Kernel Time: "
            << milliseconds << " ms\n";
  std::cout << "Estimated Bandwidth: "
            << bandwidth << " GB/s\n";
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  delete[] hostInput;
  delete[] hostOutput;

  return correct ? 0 : 1;
}