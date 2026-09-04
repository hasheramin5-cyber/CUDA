#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MATRIX_ROWS 1024
#define MATRIX_COLS 1024
#define TILE_SIZE 16
#define THREADS_X 16
#define THREADS_Y 16

__global__ void matrixTranspose(
    const float *input,
    float *output,
    int rows,
    int cols)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (x < cols && y < rows)
  {
    tile[threadIdx.y][threadIdx.x] =
        input[y * cols + x];
  }

  __syncthreads();

  int transposedX = blockIdx.y * TILE_SIZE + threadIdx.x;
  int transposedY = blockIdx.x * TILE_SIZE + threadIdx.y;

  if (transposedX < rows && transposedY < cols)
  {
    output[transposedY * rows + transposedX] =
        tile[threadIdx.x][threadIdx.y];
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

  std::vector<float> h_input(elements);
  std::vector<float> h_output(elements);

  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      h_input[row * cols + col] =
          static_cast<float>(row * cols + col);
    }
  }

  float *d_input = nullptr;
  float *d_output = nullptr;

  if (!checkCudaError(
          cudaMalloc(&d_input, bytes),
          "cudaMalloc(d_input)"))
  {
    return 1;
  }

  if (!checkCudaError(
          cudaMalloc(&d_output, bytes),
          "cudaMalloc(d_output)"))
  {
    cudaFree(d_input);
    return 1;
  }

  if (!checkCudaError(
          cudaMemcpy(
              d_input,
              h_input.data(),
              bytes,
              cudaMemcpyHostToDevice),
          "cudaMemcpy(input)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  dim3 threadsPerBlock(THREADS_X, THREADS_Y);

  dim3 blocksPerGrid(
      (cols + TILE_SIZE - 1) / TILE_SIZE,
      (rows + TILE_SIZE - 1) / TILE_SIZE);

  matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(
      d_input,
      d_output,
      rows,
      cols);

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
              bytes,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy(output)"))
  {
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  bool valid = true;

  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      float expected =
          h_input[col * cols + row];

      float actual =
          h_output[row * rows + col];

      if (actual != expected)
      {
        std::cerr << "Verification failed at row "
                  << row << ", column " << col
                  << ". Expected: " << expected
                  << ", Got: " << actual << '\n';

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
    std::cout << "Matrix transpose completed successfully.\n";
    std::cout << "Input matrix size: "
              << rows << " x " << cols << '\n';
    std::cout << "Output matrix size: "
              << cols << " x " << rows << '\n';

    std::cout << "Sample output values:\n";

    for (int i = 0; i < 5; ++i)
    {
      std::cout << "Output[" << i << "] = "
                << h_output[i] << '\n';
    }
  }
  else
  {
    std::cerr << "Output verification failed.\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);

  return valid ? 0 : 1;
}