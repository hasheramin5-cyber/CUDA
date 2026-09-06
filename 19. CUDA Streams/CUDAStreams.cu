#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NUM_ELEMENTS (1 << 20)
#define NUM_STREAMS 4
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

  std::vector<float> hostInput(NUM_ELEMENTS);
  std::vector<float> hostOutput(NUM_ELEMENTS);

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
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(
              &deviceOutput,
              bytes),
          "cudaMalloc(deviceOutput)"))
  {
    cudaFree(deviceInput);
    return 1;
  }

  cudaStream_t streams[NUM_STREAMS];

  for (int i = 0; i < NUM_STREAMS; ++i)
  {
    if (!checkCudaError(
            cudaStreamCreate(&streams[i]),
            "cudaStreamCreate"))
    {
      for (int j = 0; j < i; ++j)
      {
        cudaStreamDestroy(streams[j]);
      }

      cudaFree(deviceInput);
      cudaFree(deviceOutput);
      return 1;
    }
  }

  const int chunkSize =
      NUM_ELEMENTS / NUM_STREAMS;

  for (int i = 0; i < NUM_STREAMS; ++i)
  {
    int offset =
        i * chunkSize;

    int elements =
        (i == NUM_STREAMS - 1)
            ? NUM_ELEMENTS - offset
            : chunkSize;

    int blocks =
        (elements + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    if (!checkCudaError(
            cudaMemcpyAsync(
                deviceInput + offset,
                hostInput.data() + offset,
                elements * sizeof(float),
                cudaMemcpyHostToDevice,
                streams[i]),
            "cudaMemcpyAsync HostToDevice"))
    {
      for (int j = 0; j < NUM_STREAMS; ++j)
      {
        cudaStreamDestroy(streams[j]);
      }

      cudaFree(deviceInput);
      cudaFree(deviceOutput);
      return 1;
    }

    processKernel<<<blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
        deviceInput + offset,
        deviceOutput + offset,
        elements);

    if (!checkCudaError(
            cudaGetLastError(),
            "processKernel launch"))
    {
      for (int j = 0; j < NUM_STREAMS; ++j)
      {
        cudaStreamDestroy(streams[j]);
      }

      cudaFree(deviceInput);
      cudaFree(deviceOutput);
      return 1;
    }

    if (!checkCudaError(
            cudaMemcpyAsync(
                hostOutput.data() + offset,
                deviceOutput + offset,
                elements * sizeof(float),
                cudaMemcpyDeviceToHost,
                streams[i]),
            "cudaMemcpyAsync DeviceToHost"))
    {
      for (int j = 0; j < NUM_STREAMS; ++j)
      {
        cudaStreamDestroy(streams[j]);
      }

      cudaFree(deviceInput);
      cudaFree(deviceOutput);
      return 1;
    }
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
      cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  bool correct = true;

  for (int i = 0; i < NUM_ELEMENTS; ++i)
  {
    float expected =
        hostInput[i] * 2.0f + 1.0f;

    if (hostOutput[i] != expected)
    {
      correct = false;
      break;
    }
  }

  std::cout << "CUDA Streams\n";
  std::cout << "Number of Elements: "
            << NUM_ELEMENTS << '\n';
  std::cout << "Number of Streams: "
            << NUM_STREAMS << '\n';
  std::cout << "Result Verification: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  for (int i = 0; i < NUM_STREAMS; ++i)
  {
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return correct ? 0 : 1;
}