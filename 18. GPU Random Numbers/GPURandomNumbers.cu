#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define NUM_VALUES 1000000
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

__global__ void generateRandomNumbers(
    float *values,
    unsigned long long seed,
    int size)
{
  int index =
      blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    curandState state;

    curand_init(
        seed,
        index,
        0,
        &state);

    values[index] =
        curand_uniform(&state);
  }
}

int main()
{
  const int blocks =
      (NUM_VALUES + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;

  float *deviceValues = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceValues,
              NUM_VALUES * sizeof(float)),
          "cudaMalloc(deviceValues)"))
  {
    return 1;
  }

  generateRandomNumbers<<<blocks, THREADS_PER_BLOCK>>>(
      deviceValues,
      1234ULL,
      NUM_VALUES);

  if (!checkCudaError(
          cudaGetLastError(),
          "generateRandomNumbers launch"))
  {
    cudaFree(deviceValues);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cudaFree(deviceValues);
    return 1;
  }

  std::vector<float> hostValues(NUM_VALUES);

  if (!checkCudaError(
          cudaMemcpy(
              hostValues.data(),
              deviceValues,
              NUM_VALUES * sizeof(float),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaFree(deviceValues);
    return 1;
  }

  float minimum = hostValues[0];
  float maximum = hostValues[0];
  double sum = 0.0;

  for (int i = 0; i < NUM_VALUES; ++i)
  {
    if (hostValues[i] < minimum)
    {
      minimum = hostValues[i];
    }

    if (hostValues[i] > maximum)
    {
      maximum = hostValues[i];
    }

    sum += hostValues[i];
  }

  double average =
      sum / NUM_VALUES;

  std::cout << "GPU Random Numbers\n";
  std::cout << "Number of Values: "
            << NUM_VALUES << '\n';

  std::cout << "First 10 Values:\n";

  for (int i = 0; i < 10; ++i)
  {
    std::cout << std::fixed
              << std::setprecision(6)
              << hostValues[i] << '\n';
  }

  std::cout << "Minimum: "
            << minimum << '\n';

  std::cout << "Maximum: "
            << maximum << '\n';

  std::cout << "Average: "
            << average << '\n';

  cudaFree(deviceValues);

  return 0;
}