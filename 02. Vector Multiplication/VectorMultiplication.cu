#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define VECTOR_SIZE 1000000
#define THREADS_PER_BLOCK 256

__global__ void vectorMultiply(const float* A, const float* B, float* C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] * B[index];
    }
}

bool checkCudaError(cudaError_t error, const char* operation)
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
    const size_t bytes = size * sizeof(float);

    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);

    for (int i = 0; i < size; ++i)
    {
        h_A[i] = static_cast<float>(i + 1);
        h_B[i] = static_cast<float>((i % 10) + 1);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

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

    const int blocksPerGrid =
        (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    vectorMultiply<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_A,
        d_B,
        d_C,
        size
    );

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

    for (int i = 0; i < size; ++i)
    {
        float expected = h_A[i] * h_B[i];

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
        std::cout << "Vector multiplication completed successfully.\n";
        std::cout << "Elements processed: " << size << '\n';
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