#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>

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

bool checkSolverError(cusolverStatus_t status, const char *operation)
{
  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    std::cerr << "cuSOLVER Error during "
              << operation << '\n';
    return false;
  }

  return true;
}

int main()
{
  constexpr int N = 3;

  const size_t matrixBytes =
      N * N * sizeof(float);

  const size_t vectorBytes =
      N * sizeof(float);

  std::vector<float> hostA =
      {
          3.0f, 1.0f, 2.0f,
          1.0f, 4.0f, 1.0f,
          2.0f, 1.0f, 5.0f};

  std::vector<float> hostB =
      {
          11.0f,
          12.0f,
          19.0f};

  const std::vector<float> expected =
      {
          1.0f,
          2.0f,
          3.0f};

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceWork = nullptr;

  int *deviceIpiv = nullptr;
  int *deviceInfo = nullptr;

  cusolverDnHandle_t solverHandle = nullptr;

  if (!checkSolverError(
          cusolverDnCreate(&solverHandle),
          "cusolverDnCreate"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceA, matrixBytes),
          "cudaMalloc(deviceA)"))
  {
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceB, vectorBytes),
          "cudaMalloc(deviceB)"))
  {
    cudaFree(deviceA);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceIpiv, N * sizeof(int)),
          "cudaMalloc(deviceIpiv)"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceInfo, sizeof(int)),
          "cudaMalloc(deviceInfo)"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceIpiv);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceA,
              hostA.data(),
              matrixBytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy A"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceB,
              hostB.data(),
              vectorBytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy B"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  int workSize = 0;

  if (!checkSolverError(
          cusolverDnSgetrf_bufferSize(
              solverHandle,
              N,
              N,
              deviceA,
              N,
              &workSize),
          "cusolverDnSgetrf_bufferSize"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceWork,
              workSize * sizeof(float)),
          "cudaMalloc(deviceWork)"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkSolverError(
          cusolverDnSgetrf(
              solverHandle,
              N,
              N,
              deviceA,
              N,
              deviceWork,
              deviceIpiv,
              deviceInfo),
          "cusolverDnSgetrf"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize after LU factorization"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  int factorizationInfo = 0;

  if (!checkCudaError(
          cudaMemcpy(
              &factorizationInfo,
              deviceInfo,
              sizeof(int),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy factorization info"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (factorizationInfo != 0)
  {
    std::cerr << "LU factorization failed. Info: "
              << factorizationInfo << '\n';

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkSolverError(
          cusolverDnSgetrs(
              solverHandle,
              CUBLAS_OP_N,
              N,
              1,
              deviceA,
              N,
              deviceIpiv,
              deviceB,
              N,
              deviceInfo),
          "cusolverDnSgetrs"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize after solving"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  int solveInfo = 0;

  if (!checkCudaError(
          cudaMemcpy(
              &solveInfo,
              deviceInfo,
              sizeof(int),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy solve info"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  if (solveInfo != 0)
  {
    std::cerr << "Linear system solve failed. Info: "
              << solveInfo << '\n';

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  std::vector<float> hostX(N);

  if (!checkCudaError(
          cudaMemcpy(
              hostX.data(),
              deviceB,
              vectorBytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy solution"))
  {
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceWork);
    cudaFree(deviceIpiv);
    cudaFree(deviceInfo);
    cusolverDnDestroy(solverHandle);
    return 1;
  }

  bool correct = true;

  for (int i = 0; i < N; ++i)
  {
    if (std::fabs(hostX[i] - expected[i]) > 1e-5f)
    {
      correct = false;
      break;
    }
  }

  std::cout << "Linear System Solver\n";
  std::cout << "Matrix Size: "
            << N << " x " << N << '\n';

  std::cout << "Solution: ";

  for (float value : hostX)
  {
    std::cout << value << ' ';
  }

  std::cout << '\n';

  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceWork);
  cudaFree(deviceIpiv);
  cudaFree(deviceInfo);

  cusolverDnDestroy(solverHandle);

  return correct ? 0 : 1;
}