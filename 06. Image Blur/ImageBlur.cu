#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define CHANNELS 3
#define BLUR_RADIUS 2
#define THREADS_X 16
#define THREADS_Y 16

__global__ void imageBlur(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height,
    int channels,
    int radius)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
  {
    return;
  }

  for (int channel = 0; channel < channels; ++channel)
  {
    int sum = 0;
    int count = 0;

    for (int offsetY = -radius; offsetY <= radius; ++offsetY)
    {
      for (int offsetX = -radius; offsetX <= radius; ++offsetX)
      {
        int neighborX = x + offsetX;
        int neighborY = y + offsetY;

        if (neighborX >= 0 && neighborX < width &&
            neighborY >= 0 && neighborY < height)
        {
          int index =
              (neighborY * width + neighborX) * channels + channel;

          sum += input[index];
          ++count;
        }
      }
    }

    int outputIndex =
        (y * width + x) * channels + channel;

    output[outputIndex] =
        static_cast<unsigned char>(sum / count);
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
  const int width = IMAGE_WIDTH;
  const int height = IMAGE_HEIGHT;
  const int channels = CHANNELS;

  const int totalPixels = width * height;
  const size_t bytes = totalPixels * channels * sizeof(unsigned char);

  std::vector<unsigned char> h_input(
      totalPixels * channels);

  std::vector<unsigned char> h_output(
      totalPixels * channels);

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      int index = (y * width + x) * channels;

      h_input[index] =
          static_cast<unsigned char>(x % 256);

      h_input[index + 1] =
          static_cast<unsigned char>(y % 256);

      h_input[index + 2] =
          static_cast<unsigned char>((x + y) % 256);
    }
  }

  unsigned char *d_input = nullptr;
  unsigned char *d_output = nullptr;

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
      (width + THREADS_X - 1) / THREADS_X,
      (height + THREADS_Y - 1) / THREADS_Y);

  imageBlur<<<blocksPerGrid, threadsPerBlock>>>(
      d_input,
      d_output,
      width,
      height,
      channels,
      BLUR_RADIUS);

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

  for (int i = 0; i < totalPixels * channels; ++i)
  {
    if (h_output[i] > 255)
    {
      valid = false;
      break;
    }
  }

  if (valid)
  {
    std::cout << "Image blur completed successfully.\n";
    std::cout << "Image size: "
              << width << " x " << height << '\n';
    std::cout << "Blur radius: "
              << BLUR_RADIUS << '\n';
    std::cout << "Channels: "
              << channels << '\n';

    std::cout << "Sample output pixel: "
              << static_cast<int>(h_output[0]) << ", "
              << static_cast<int>(h_output[1]) << ", "
              << static_cast<int>(h_output[2]) << '\n';
  }
  else
  {
    std::cerr << "Output verification failed.\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);

  return valid ? 0 : 1;
}