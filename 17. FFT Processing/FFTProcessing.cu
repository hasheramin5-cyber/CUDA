#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <cmath>

#define SIGNAL_SIZE 4096

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

bool checkCufftError(
    cufftResult result,
    const char *operation)
{
  if (result != CUFFT_SUCCESS)
  {
    std::cerr << "cuFFT Error during "
              << operation << '\n';
    return false;
  }

  return true;
}

int main()
{
  std::vector<cufftComplex> hostSignal(SIGNAL_SIZE);

  for (int i = 0; i < SIGNAL_SIZE; ++i)
  {
    float value =
        std::sin(2.0f * 3.14159265359f * 50.0f * i / SIGNAL_SIZE) +
        0.5f * std::sin(
                   2.0f * 3.14159265359f * 120.0f * i / SIGNAL_SIZE);

    hostSignal[i].x = value;
    hostSignal[i].y = 0.0f;
  }

  cufftComplex *deviceSignal = nullptr;

  if (!checkCudaError(
          cudaMalloc(
              &deviceSignal,
              SIGNAL_SIZE * sizeof(cufftComplex)),
          "cudaMalloc(deviceSignal)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              deviceSignal,
              hostSignal.data(),
              SIGNAL_SIZE * sizeof(cufftComplex),
              cudaMemcpyHostToDevice),
          "cudaMemcpy HostToDevice"))
  {
    cudaFree(deviceSignal);
    return 1;
  }

  cufftHandle plan;

  if (!checkCufftError(
          cufftPlan1d(
              &plan,
              SIGNAL_SIZE,
              CUFFT_C2C,
              1),
          "cufftPlan1d"))
  {
    cudaFree(deviceSignal);
    return 1;
  }

  if (!checkCufftError(
          cufftExecC2C(
              plan,
              deviceSignal,
              deviceSignal,
              CUFFT_FORWARD),
          "cufftExecC2C"))
  {
    cufftDestroy(plan);
    cudaFree(deviceSignal);
    return 1;
  }

  if (!checkCudaError(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize"))
  {
    cufftDestroy(plan);
    cudaFree(deviceSignal);
    return 1;
  }

  std::vector<cufftComplex> hostResult(SIGNAL_SIZE);

  if (!checkCudaError(
          cudaMemcpy(
              hostResult.data(),
              deviceSignal,
              SIGNAL_SIZE * sizeof(cufftComplex),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy DeviceToHost"))
  {
    cufftDestroy(plan);
    cudaFree(deviceSignal);
    return 1;
  }

  int strongestFrequency = 0;
  float strongestMagnitude = 0.0f;

  for (int i = 1; i < SIGNAL_SIZE / 2; ++i)
  {
    float real = hostResult[i].x;
    float imaginary = hostResult[i].y;

    float magnitude =
        std::sqrt(
            real * real +
            imaginary * imaginary);

    if (magnitude > strongestMagnitude)
    {
      strongestMagnitude = magnitude;
      strongestFrequency = i;
    }
  }

  std::cout << "FFT Processing\n";
  std::cout << "Signal Size: "
            << SIGNAL_SIZE << '\n';
  std::cout << "Strongest Frequency Bin: "
            << strongestFrequency << '\n';
  std::cout << "Magnitude: "
            << strongestMagnitude << '\n';

  cufftDestroy(plan);
  cudaFree(deviceSignal);

  return 0;
}