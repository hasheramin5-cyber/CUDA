#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MATRIX_SIZE 512
#define THREADS_X 16
#define THREADS_Y 16

__global__ void matrixMultiply(
    const float *A,
    const float *B,
    float *C,
    int size)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size && col < size)
  {
    float sum = 0.0f;

    for (int k = 0; k < size; ++k)
    {
      sum += A[row * size + k] * B[k * size + col];
    }

    C[row * size + col] = sum;
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
  const int size = MATRIX_SIZE;
  const int elements = size * size;
  const size_t bytes = elements * sizeof(float);

  std::vector<float> h_A(elements);
  std::vector<float> h_B(elements);
  std::vector<float> h_C(elements);

  for (int i = 0; i < elements; ++i)
  {
    h_A[i] = static_cast<float>((i % 10) + 1);
    h_B[i] = static_cast<float>((i % 5) + 1);
  }

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  if (!checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc(d_A)"))
  {
    return 1;
  }

  if (!checkCudaError(cudaMalloc(&d_B, bytes), "cudaMalloc(d_B)"))
  {
    cudaFree(d_A);
    return 1;
  }

  if (!checkCudaError(cudaMalloc(&d_C, bytes), "cudaMalloc(d_C)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy(A)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy(B)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  dim3 threadsPerBlock(THREADS_X, THREADS_Y);

  dim3 blocksPerGrid(
      (size + THREADS_X - 1) / THREADS_X,
      (size + THREADS_Y - 1) / THREADS_Y);

  matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(
      d_A,
      d_B,
      d_C,
      size);

  if (!checkCudaError(cudaGetLastError(), "kernel launch"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(cudaDeviceSynchronize(), "kernel execution"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(C)"))
  {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 1;
  }

  bool valid = true;

  for (int row = 0; row < size; ++row)
  {
    for (int col = 0; col < size; ++col)
    {
      float expected = 0.0f;

      for (int k = 0; k < size; ++k)
      {
        expected +=
            h_A[row * size + k] *
            h_B[k * size + col];
      }

      if (h_C[row * size + col] != expected)
      {
        std::cerr << "Verification failed at row "
                  << row << ", column " << col
                  << ". Expected: " << expected
                  << ", Got: "
                  << h_C[row * size + col] << '\n';

        valid = false;
        break;
      }
    }

    if (!valid)
    {
      break;
    }
  }

  if (valid)
  {
    std::cout << "Matrix multiplication completed successfully.\n";
    std::cout << "Matrix size: "
              << size << " x " << size << '\n';
    std::cout << "Sample results:\n";

    for (int i = 0; i < 5; ++i)
    {
      std::cout << "C[" << i << "] = "
                << h_C[i] << '\n';
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return valid ? 0 : 1;
}