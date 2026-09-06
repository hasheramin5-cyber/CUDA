#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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
          cudaMalloc(&deviceInput, bytes),
          "cudaMalloc(deviceInput)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&deviceOutput, bytes),
          "cudaMalloc(deviceOutput)"))
  {
    cudaFree(deviceInput);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceInput,
              hostInput.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  cudaStream_t stream;

  if (!checkCudaError(
          cudaStreamCreate(&stream),
          "cudaStreamCreate"))
  {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  cudaGraph_t graph;

  if (!checkCudaError(
          cudaGraphCreate(&graph, 0),
          "cudaGraphCreate"))
  {
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  cudaGraphNode_t kernelNode;

  cudaKernelNodeParams kernelParams{};

  int size = NUM_ELEMENTS;

  void *kernelArguments[] =
      {
          &deviceInput,
          &deviceOutput,
          &size};

  kernelParams.func =
      reinterpret_cast<void *>(processKernel);

  kernelParams.gridDim =
      dim3(
          (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) /
          THREADS_PER_BLOCK);

  kernelParams.blockDim =
      dim3(THREADS_PER_BLOCK);

  kernelParams.sharedMemBytes = 0;
  kernelParams.kernelParams = kernelArguments;

  if (!checkCudaError(
          cudaGraphAddKernelNode(
              &kernelNode,
              graph,
              nullptr,
              0,
              &kernelParams),
          "cudaGraphAddKernelNode"))
  {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  cudaGraphExec_t graphExec;

  if (!checkCudaError(
          cudaGraphInstantiate(
              &graphExec,
              graph,
              nullptr,
              nullptr,
              0),
          "cudaGraphInstantiate"))
  {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaGraphLaunch(
              graphExec,
              stream),
          "cudaGraphLaunch"))
  {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaStreamSynchronize(stream),
          "cudaStreamSynchronize"))
  {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              hostOutput.data(),
              deviceOutput,
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
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

  std::cout << "CUDA Graphs\n";
  std::cout << "Number of Elements: "
            << NUM_ELEMENTS << '\n';
  std::cout << "Graph Execution: "
            << (correct ? "PASSED" : "FAILED")
            << '\n';

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return correct ? 0 : 1;
}