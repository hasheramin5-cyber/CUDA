#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

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

void printMemorySize(size_t bytes)
{
  double megabytes =
      static_cast<double>(bytes) / (1024.0 * 1024.0);

  double gigabytes =
      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(2)
            << megabytes << " MB ("
            << gigabytes << " GB)";
}

int main()
{
  int deviceCount = 0;

  if (!checkCudaError(
          cudaGetDeviceCount(&deviceCount),
          "cudaGetDeviceCount"))
  {
    return 1;
  }

  if (deviceCount == 0)
  {
    std::cerr << "No CUDA-capable NVIDIA GPU was detected.\n";
    return 1;
  }

  std::cout << "CUDA Device Query\n";
  std::cout << "CUDA-capable devices: "
            << deviceCount << "\n\n";

  for (int device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp properties{};

    if (!checkCudaError(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties"))
    {
      return 1;
    }

    std::cout << "Device " << device << '\n';

    std::cout << "Name: "
              << properties.name << '\n';

    std::cout << "Compute Capability: "
              << properties.major << "."
              << properties.minor << '\n';

    std::cout << "Total Global Memory: ";
    printMemorySize(properties.totalGlobalMem);
    std::cout << '\n';

    std::cout << "Shared Memory per Block: ";
    printMemorySize(properties.sharedMemPerBlock);
    std::cout << '\n';

    std::cout << "Registers per Block: "
              << properties.regsPerBlock << '\n';

    std::cout << "Warp Size: "
              << properties.warpSize << '\n';

    std::cout << "Maximum Threads per Block: "
              << properties.maxThreadsPerBlock << '\n';

    std::cout << "Maximum Threads in X Dimension: "
              << properties.maxThreadsDim[0] << '\n';

    std::cout << "Maximum Threads in Y Dimension: "
              << properties.maxThreadsDim[1] << '\n';

    std::cout << "Maximum Threads in Z Dimension: "
              << properties.maxThreadsDim[2] << '\n';

    std::cout << "Maximum Grid Size X: "
              << properties.maxGridSize[0] << '\n';

    std::cout << "Maximum Grid Size Y: "
              << properties.maxGridSize[1] << '\n';

    std::cout << "Maximum Grid Size Z: "
              << properties.maxGridSize[2] << '\n';

    std::cout << "Multiprocessors: "
              << properties.multiProcessorCount << '\n';

    std::cout << "Clock Rate: "
              << properties.clockRate / 1000
              << " MHz\n";

    std::cout << "Memory Clock Rate: "
              << properties.memoryClockRate / 1000
              << " MHz\n";

    std::cout << "Memory Bus Width: "
              << properties.memoryBusWidth
              << " bits\n";

    std::cout << "Concurrent Kernels: "
              << (properties.concurrentKernels ? "Yes" : "No")
              << '\n';

    std::cout << "Unified Addressing: "
              << (properties.unifiedAddressing ? "Yes" : "No")
              << '\n';

    std::cout << "Managed Memory: "
              << (properties.managedMemory ? "Yes" : "No")
              << '\n';

    std::cout << '\n';
  }

  int currentDevice = 0;

  if (!checkCudaError(
          cudaGetDevice(&currentDevice),
          "cudaGetDevice"))
  {
    return 1;
  }

  std::cout << "Current CUDA Device: "
            << currentDevice << '\n';

  return 0;
}