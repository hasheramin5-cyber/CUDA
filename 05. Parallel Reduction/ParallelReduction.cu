#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define VECTOR_SIZE 1048576
#define THREADS_PER_BLOCK 256

__global__ void reduceSum(const float *input, float *output, int size)
{
  __shared__ float sharedData[THREADS_PER_BLOCK];

  int threadId = threadIdx.x;
  int globalId = blockIdx.x * blockDim.x + threadId;

  sharedData[threadId] = (globalId < size) ? input[globalId] : 0.0f;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (threadId < stride)
    {
      sharedData[threadId] += sharedData[threadId + stride];
    }

    __syncthreads();
  }

  if (threadId == 0)
  {
    output[blockIdx.x] = sharedData[0];
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

int main()
{
  const int size = VECTOR_SIZE;
  const size_t inputBytes = size * sizeof(float);

  const int blocksPerGrid =
      (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const size_t outputBytes = blocksPerGrid * sizeof(float);

  std::vector<float> h_input(size);
  std::vector<float> h_output(blocksPerGrid);

  for (int i = 0; i < size; ++i)
  {
    h_input[i] = 1.0f;
  }

  float *d_input = nullptr;
  float *d_output = nullptr;

  if (!checkCudaError(
          cudaMalloc(&d_input, inputBytes),
          "cudaMalloc(d_input)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&d_output, outputBytes),
          "cudaMalloc(d_output)"))
  {
    cudaFree(d_input);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              d_input,
              h_input.data(),
              inputBytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy(input)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  reduceSum<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
      d_input,
      d_output,
      size);

  if (!checkCudaError(
          cudaGetLastError(),
          "kernel launch"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "kernel execution"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              h_output.data(),
              d_output,
              outputBytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy(output)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  float gpuSum = 0.0f;

  for (float value : h_output)
  {
    gpuSum += value;
  }

  float cpuSum = 0.0f;

  for (float value : h_input)
  {
    cpuSum += value;
  }

  bool valid = (gpuSum == cpuSum);

  if (valid)
  {
    std::cout << "Parallel reduction completed successfully.\n";
    std::cout << "Elements processed: " << size << '\n';
    std::cout << "GPU sum: " << gpuSum << '\n';
    std::cout << "CPU sum: " << cpuSum << '\n';
  }
  else
  {
    std::cerr << "Verification failed.\n";
    std::cerr << "GPU sum: " << gpuSum << '\n';
    std::cerr << "CPU sum: " << cpuSum << '\n';
  }

  cudaFree(d_input);
  cudaFree(d_output);

  return valid ? 0 : 1;
}