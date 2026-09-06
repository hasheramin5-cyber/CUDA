#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <cmath>

#define ROWS 4
#define COLS 5
#define NNZ 7

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

bool checkCusparseError(
    cusparseStatus_t status,
    const char *operation)
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    std::cerr << "cuSPARSE Error during "
              << operation << '\n';
    return false;
  }

  return true;
}

int main()
{
  std::vector<int> hostRowOffsets =
      {
          0, 2, 4, 5, 7};

  std::vector<int> hostColumnIndices =
      {
          0, 3,
          1, 4,
          2,
          0, 4};

  std::vector<float> hostValues =
      {
          1.0f, 2.0f,
          3.0f, 4.0f,
          5.0f,
          6.0f, 7.0f};

  std::vector<float> hostVector =
      {
          1.0f,
          2.0f,
          3.0f,
          4.0f,
          5.0f};

  std::vector<float> hostResult(ROWS, 0.0f);
  std::vector<float> expected =
      {
          9.0f,
          26.0f,
          15.0f,
          41.0f};

  int *deviceRowOffsets = nullptr;
  int *deviceColumnIndices = nullptr;
  float *deviceValues = nullptr;
  float *deviceVector = nullptr;
  float *deviceResult = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceRowOffsets,
              (ROWS + 1) * sizeof(int)),
          "cudaMalloc(rowOffsets)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceColumnIndices,
              NNZ * sizeof(int)),
          "cudaMalloc(columnIndices)"))
  {
    cudaFree(deviceRowOffsets);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceValues,
              NNZ * sizeof(float)),
          "cudaMalloc(values)"))
  {
    cudaFree(deviceRowOffsets);
    cudaFree(deviceColumnIndices);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceVector,
              COLS * sizeof(float)),
          "cudaMalloc(vector)"))
  {
    cudaFree(deviceRowOffsets);
    cudaFree(deviceColumnIndices);
    cudaFree(deviceValues);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceResult,
              ROWS * sizeof(float)),
          "cudaMalloc(result)"))
  {
    cudaFree(deviceRowOffsets);
    cudaFree(deviceColumnIndices);
    cudaFree(deviceValues);
    cudaFree(deviceVector);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceRowOffsets,
              hostRowOffsets.data(),
              (ROWS + 1) * sizeof(int),
              cudaMemcpyHostToDevice),
          "cudaMemcpy rowOffsets"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceColumnIndices,
              hostColumnIndices.data(),
              NNZ * sizeof(int),
              cudaMemcpyHostToDevice),
          "cudaMemcpy columnIndices"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceValues,
              hostValues.data(),
              NNZ * sizeof(float),
              cudaMemcpyHostToDevice),
          "cudaMemcpy values"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceVector,
              hostVector.data(),
              COLS * sizeof(float),
              cudaMemcpyHostToDevice),
          "cudaMemcpy vector"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMemset(
              deviceResult,
              0,
              ROWS * sizeof(float)),
          "cudaMemset result"))
  {
    return 1;
  }

  cusparseHandle_t handle;

  if (!checkCusparseError(
          cusparseCreate(&handle),
          "cusparseCreate"))
  {
    return 1;
  }

  cusparseSpMatDescr_t matrix;
  cusparseDnVecDescr_t vector;
  cusparseDnVecDescr_t result;

  if (!checkCusparseError(
          cusparseCreateCsr(
              &matrix,
              ROWS,
              COLS,
              NNZ,
              deviceRowOffsets,
              deviceColumnIndices,
              deviceValues,
              CUSPARSE_INDEX_32I,
              CUSPARSE_INDEX_32I,
              CUSPARSE_INDEX_BASE_ZERO,
              CUDA_R_32F),
          "cusparseCreateCsr"))
  {
    cusparseDestroy(handle);
    return 1;
  }

  if (!checkCusparseError(
          cusparseCreateDnVec(
              &vector,
              COLS,
              deviceVector,
              CUDA_R_32F),
          "cusparseCreateDnVec(vector)"))
  {
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  if (!checkCusparseError(
          cusparseCreateDnVec(
              &result,
              ROWS,
              deviceResult,
              CUDA_R_32F),
          "cusparseCreateDnVec(result)"))
  {
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  size_t bufferSize = 0;

  if (!checkCusparseError(
          cusparseSpMV_bufferSize(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha,
              matrix,
              vector,
              &beta,
              result,
              CUDA_R_32F,
              CUSPARSE_SPMV_ALG_DEFAULT,
              &bufferSize),
          "cusparseSpMV_bufferSize"))
  {
    cusparseDestroyDnVec(result);
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  void *buffer = nullptr;

  if (!checkCudaError(
          cudaMalloc(&buffer, bufferSize),
          "cudaMalloc(buffer)"))
  {
    cusparseDestroyDnVec(result);
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  if (!checkCusparseError(
          cusparseSpMV(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha,
              matrix,
              vector,
              &beta,
              result,
              CUDA_R_32F,
              CUSPARSE_SPMV_ALG_DEFAULT,
              buffer),
          "cusparseSpMV"))
  {
    cudaFree(buffer);
    cusparseDestroyDnVec(result);
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(buffer);
    cusparseDestroyDnVec(result);
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostResult.data(),
              deviceResult,
              ROWS * sizeof(float),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy result"))
  {
    cudaFree(buffer);
    cusparseDestroyDnVec(result);
    cusparseDestroyDnVec(vector);
    cusparseDestroySpMat(matrix);
    cusparseDestroy(handle);
    return 1;
  }

  bool correct = true;

  for (int i = 0; i < ROWS; ++i)
  {
    if (std::fabs(hostResult[i] - expected[i]) > 0.001f)
    {
      correct = false;
      break;
    }
  }

  std::cout << "Sparse Matrix Operations\n";
  std::cout << "Matrix Size: "
            << ROWS << " x " << COLS << '\n';
  std::cout << "Non-Zero Elements: "
            << NNZ << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaFree(buffer);

  cusparseDestroyDnVec(result);
  cusparseDestroyDnVec(vector);
  cusparseDestroySpMat(matrix);
  cusparseDestroy(handle);

  cudaFree(deviceRowOffsets);
  cudaFree(deviceColumnIndices);
  cudaFree(deviceValues);
  cudaFree(deviceVector);
  cudaFree(deviceResult);

  return correct ? 0 : 1;
}