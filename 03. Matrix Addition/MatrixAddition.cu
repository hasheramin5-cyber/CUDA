#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MATRIX_ROWS 1024
#define MATRIX_COLS 1024
#define THREADS_X 16
#define THREADS_Y 16

__global__ void matrixAdd(
    const float *A,
    const float *B,
    float *C,
    int rows,
    int cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols)
  {
    int index = row * cols + col;
    C[index] = A[index] + B[index];
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
  const int rows = MATRIX_ROWS;
  const int cols = MATRIX_COLS;
  const int elements = rows * cols;
  const size_t bytes = elements * sizeof(float);

  std::vector<float> h_A(elements);
  std::vector<float> h_B(elements);
  std::vector<float> h_C(elements);

  for (int i = 0; i < elements; ++i)
  {
    h_A[i] = static_cast<float>(i % 100);
    h_B[i] = static_cast<float>((i % 50) + 1);
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
      (cols + THREADS_X - 1) / THREADS_X,
      (rows + THREADS_Y - 1) / THREADS_Y);

  matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(
      d_A,
      d_B,
      d_C,
      rows,
      cols);

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

  for (int i = 0; i < elements; ++i)
  {
    float expected = h_A[i] + h_B[i];

    if (h_C[i] != expected)
    {
      std::cerr << "Verification failed at index " << i
                << ". Expected: " << expected
                << ", Got: " << h_C[i] << '\n';

      valid = false;
      break;
    }
  }

  if (valid)
  {
    std::cout << "Matrix addition completed successfully.\n";
    std::cout << "Matrix size: "
              << rows << " x " << cols << '\n';
    std::cout << "Elements processed: "
              << elements << '\n';

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