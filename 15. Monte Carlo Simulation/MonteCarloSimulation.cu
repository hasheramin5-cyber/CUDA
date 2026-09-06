#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>
#include <vector>

#define NUM_SAMPLES 10000000
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

__global__ void setupRandomStates(
    curandState *states,
    unsigned long long seed,
    int size)
{
  int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    curand_init(seed, index, 0, &states[index]);
  }
}

__global__ void monteCarloKernel(
    curandState *states,
    unsigned int *results,
    int size)
{
  int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    curandState localState = states[index];

    float x = curand_uniform(&localState);
    float y = curand_uniform(&localState);

    float distance =
        x * x + y * y;

    results[index] =
        (distance <= 1.0f) ? 1 : 0;

    states[index] = localState;
  }
}

int main()
{
  const int blocks =
      (NUM_SAMPLES + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  curandState *deviceStates = nullptr;
  unsigned int *deviceResults = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceStates,
              NUM_SAMPLES * sizeof(curandState)),
          "cudaMalloc(deviceStates)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceResults,
              NUM_SAMPLES * sizeof(unsigned int)),
          "cudaMalloc(deviceResults)"))
  {
    cudaFree(deviceStates);
    return 1;
  }

  setupRandomStates<<<blocks, THREADS_PER_BLOCK>>>(
      deviceStates,
      1234ULL,
      NUM_SAMPLES);

  if (!checkCudaError(
          cudaGetLastError(),
          "setupRandomStates launch"))
  {
    cudaFree(deviceStates);
    cudaFree(deviceResults);
    return 1;
  }

  monteCarloKernel<<<blocks, THREADS_PER_BLOCK>>>(
      deviceStates,
      deviceResults,
      NUM_SAMPLES);

  if (!checkCudaError(
          cudaGetLastError(),
          "monteCarloKernel launch"))
  {
    cudaFree(deviceStates);
    cudaFree(deviceResults);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceStates);
    cudaFree(deviceResults);
    return 1;
  }

  std::vector<unsigned int> hostResults(NUM_SAMPLES);

  if (!checkCudaError(
          cudaMemcpy(
              hostResults.data(),
              deviceResults,
              NUM_SAMPLES * sizeof(unsigned int),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaFree(deviceStates);
    cudaFree(deviceResults);
    return 1;
  }

  unsigned long long pointsInside = 0;

  for (int i = 0; i < NUM_SAMPLES; ++i)
  {
    pointsInside += hostResults[i];
  }

  double piEstimate =
      4.0 * static_cast<double>(pointsInside) /
      static_cast<double>(NUM_SAMPLES);

  std::cout << "Monte Carlo Simulation\n";
  std::cout << "Number of Samples: "
            << NUM_SAMPLES << '\n';
  std::cout << "Points Inside Circle: "
            << pointsInside << '\n';
  std::cout << "Estimated Value of Pi: "
            << std::fixed << std::setprecision(6)
            << piEstimate << '\n';

  cudaFree(deviceStates);
  cudaFree(deviceResults);

  return 0;
}