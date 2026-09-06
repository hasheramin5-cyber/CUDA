#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#define MATRIX_SIZE 1024

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

bool checkCublasError(
    cublasStatus_t status,
    const char *operation)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "cuBLAS Error during "
              << operation << '\n';
    return false;
  }

  return true;
}

int main()
{
  const int rows = MATRIX_SIZE;
  const int columns = MATRIX_SIZE;
  const int elements = rows * columns;

  const size_t bytes =
      static_cast<size_t>(elements) * sizeof(float);

  std::vector<float> hostA(elements);
  std::vector<float> hostB(elements);
  std::vector<float> hostC(elements, 0.0f);
  std::vector<float> expected(elements, 0.0f);

  for (int i = 0; i < elements; ++i)
  {
    hostA[i] = 1.0f;
    hostB[i] = 2.0f;
  }

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  if (!checkCudaError(
          cudaMalloc(&deviceA, bytes),
          "cudaMalloc(deviceA)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceB, bytes),
          "cudaMalloc(deviceB)"))
  {
    cudaFree(deviceA);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceC, bytes),
          "cudaMalloc(deviceC)"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceA,
              hostA.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy A"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceB,
              hostB.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy B"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  cublasHandle_t handle;

  if (!checkCublasError(
          cublasCreate(&handle),
          "cublasCreate"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  if (!checkCublasError(
          cublasSgemm(
              handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              rows,
              columns,
              columns,
              &alpha,
              deviceA,
              rows,
              deviceB,
              columns,
              &beta,
              deviceC,
              rows),
          "cublasSgemm"))
  {
    cublasDestroy(handle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cublasDestroy(handle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostC.data(),
              deviceC,
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy C"))
  {
    cublasDestroy(handle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return 1;
  }

  const float expectedValue =
      static_cast<float>(MATRIX_SIZE * 2);

  bool correct = true;

  for (int i = 0; i < elements; ++i)
  {
    if (std::fabs(hostC[i] - expectedValue) > 0.01f)
    {
      correct = false;
      break;
    }
  }

  std::cout << "cuBLAS Matrix Multiplication\n";
  std::cout << "Matrix Size: "
            << MATRIX_SIZE << " x "
            << MATRIX_SIZE << '\n';
  std::cout << "Expected Value: "
            << expectedValue << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cublasDestroy(handle);
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  return correct ? 0 : 1;
}