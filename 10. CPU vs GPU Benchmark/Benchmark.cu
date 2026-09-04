#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#define VECTOR_SIZE 10000000
#define THREADS_PER_BLOCK 256

__global__ void vectorAddGPU(
    const float *A,
    const float *B,
    float *C,
    int size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    C[index] = A[index] + B[index];
  }
}

void vectorAddCPU(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C)
{
  for (size_t i = 0; i < A.size(); ++i)
  {
    C[i] = A[i] + B[i];
  }
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

bool verifyResults(
    const std::vector<float> &cpuResult,
    const std::vector<float> &gpuResult)
{
  if (cpuResult.size() != gpuResult.size())
  {
    return false;
  }

  for (size_t i = 0; i < cpuResult.size(); ++i)
  {
    if (std::fabs(cpuResult[i] - gpuResult[i]) > 1e-5f)
    {
      std::cerr << "Verification failed at index "
                << i << ". CPU: "
                << cpuResult[i]
                << ", GPU: "
                << gpuResult[i] << '\n';

      return false;
    }
  }

  return true;
}

int main()
{
  const int size = VECTOR_SIZE;
  const size_t bytes = size * sizeof(float);

  std::vector<float> h_A(size);
  std::vector<float> h_B(size);
  std::vector<float> h_cpuResult(size);
  std::vector<float> h_gpuResult(size);

  for (int i = 0; i < size; ++i)
  {
    h_A[i] = static_cast<float>(i % 100);
    h_B[i] = static_cast<float>((i % 50) + 1);
  }

  auto cpuStart = std::chrono::high_resolution_clock::now();

  vectorAddCPU(
      h_A,
      h_B,
      h_cpuResult);

  auto cpuEnd = std::chrono::high_resolution_clock::now();

  double cpuTime =
      std::chrono::duration<double, std::milli>(
          cpuEnd - cpuStart)
          .count();

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  if (!checkCudaError(
          cudaMalloc(&d_A, bytes),
          "cudaMalloc(d_A)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&d_B, bytes),
          "cudaMalloc(d_B)"))
  {
    cudaFree(d_A);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&d_C, bytes),
          "cudaMalloc(d_C)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              d_A,
              h_A.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy(A)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              d_B,
              h_B.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy(B)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  const int blocksPerGrid =
      (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  if (!checkCudaError(
          cudaEventCreate(&startEvent),
          "cudaEventCreate(startEvent)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaEventCreate(&stopEvent),
          "cudaEventCreate(stopEvent)"))
  {
    cudaEventDestroy(startEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaEventRecord(startEvent),
          "cudaEventRecord(startEvent)"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  vectorAddGPU<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
      d_A,
      d_B,
      d_C,
      size);

  if (!checkCudaError(
          cudaGetLastError(),
          "kernel launch"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaEventRecord(stopEvent),
          "cudaEventRecord(stopEvent)"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaEventSynchronize(stopEvent),
          "cudaEventSynchronize(stopEvent)"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  float gpuTime = 0.0f;

  if (!checkCudaError(
          cudaEventElapsedTime(
              &gpuTime,
              startEvent,
              stopEvent),
          "cudaEventElapsedTime"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              h_gpuResult.data(),
              d_C,
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy(C)"))
  {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  bool valid = verifyResults(
      h_cpuResult,
      h_gpuResult);

  if (valid)
  {
    std::cout << "CPU vs GPU benchmark completed successfully.\n";
    std::cout << "Elements processed: "
              << size << '\n';
    std::cout << "CPU execution time: "
              << cpuTime << " ms\n";
    std::cout << "GPU kernel execution time: "
              << gpuTime << " ms\n";

    if (gpuTime > 0.0f)
    {
      std::cout << "GPU speedup: "
                << cpuTime / gpuTime
                << "x\n";
    }

    std::cout << "Results verified successfully.\n";
  }

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return valid ? 0 : 1;
}