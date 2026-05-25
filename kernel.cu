#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kVectorSize = 1'000'000;
constexpr float kScale = 2.5f;
constexpr int kImageWidth = 1024;
constexpr int kImageHeight = 1024;
constexpr unsigned char kThreshold = 128;

void checkCuda(cudaError_t result, const char* call, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << call << " at " << file << ':' << line
                  << " -> " << cudaGetErrorString(result) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

template <typename Callable>
double measureCpuMilliseconds(Callable&& callable) {
    const auto start = std::chrono::high_resolution_clock::now();
    callable();
    const auto finish = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

bool compareVectors(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    const float epsilon = 1.0e-5f;

    for (int i = 0; i < static_cast<int>(cpu.size()); ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > epsilon) {
            std::cerr << "Vector mismatch at index " << i << ": CPU = " << cpu[i]
                      << ", GPU = " << gpu[i] << '\n';
            return false;
        }
    }

    return true;
}

bool compareImages(const std::vector<unsigned char>& cpu,
                   const std::vector<unsigned char>& gpu) {
    for (int i = 0; i < static_cast<int>(cpu.size()); ++i) {
        if (cpu[i] != gpu[i]) {
            std::cerr << "Image mismatch at index " << i
                      << ": CPU = " << static_cast<int>(cpu[i])
                      << ", GPU = " << static_cast<int>(gpu[i]) << '\n';
            return false;
        }
    }

    return true;
}

}  // namespace

void scaleVectorCPU(const float* A, float* B, int N, float k) {
    for (int i = 0; i < N; ++i) {
        B[i] = A[i] * k;
    }
}

__global__ void scaleVectorCUDA(const float* A, float* B, int N, float k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        B[idx] = A[idx] * k;
    }
}

void thresholdFilterCPU(unsigned char* input,
                        unsigned char* output,
                        int width,
                        int height,
                        unsigned char threshold) {
    const int totalPixels = width * height;

    for (int i = 0; i < totalPixels; ++i) {
        output[i] = input[i] > threshold ? 255 : 0;
    }
}

__global__ void thresholdFilterCUDA(unsigned char* input,
                                    unsigned char* output,
                                    int width,
                                    int height,
                                    unsigned char threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int idx = y * width + x;
        output[idx] = input[idx] > threshold ? 255 : 0;
    }
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    std::cout << "Lab 4, variant 3\n";
    std::cout << "CUDA device: " << deviceProperties.name << "\n\n";

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> floatDistribution(0.0f, 100.0f);
    std::uniform_int_distribution<int> byteDistribution(0, 255);

    std::vector<float> h_vectorA(kVectorSize);
    std::vector<float> h_vectorCpu(kVectorSize);
    std::vector<float> h_vectorGpu(kVectorSize);

    for (float& value : h_vectorA) {
        value = floatDistribution(generator);
    }

    const double vectorCpuTimeMs = measureCpuMilliseconds([&]() {
        scaleVectorCPU(h_vectorA.data(), h_vectorCpu.data(), kVectorSize, kScale);
    });

    float* d_vectorA = nullptr;
    float* d_vectorB = nullptr;
    const size_t vectorBytes = static_cast<size_t>(kVectorSize) * sizeof(float);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vectorA), vectorBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vectorB), vectorBytes));
    CUDA_CHECK(cudaMemcpy(d_vectorA,
                          h_vectorA.data(),
                          vectorBytes,
                          cudaMemcpyHostToDevice));

    const int vectorThreadsPerBlock = 256;
    const int vectorBlocks = (kVectorSize + vectorThreadsPerBlock - 1) /
                             vectorThreadsPerBlock;

    cudaEvent_t vectorStart = nullptr;
    cudaEvent_t vectorStop = nullptr;
    CUDA_CHECK(cudaEventCreate(&vectorStart));
    CUDA_CHECK(cudaEventCreate(&vectorStop));
    CUDA_CHECK(cudaEventRecord(vectorStart));
    scaleVectorCUDA<<<vectorBlocks, vectorThreadsPerBlock>>>(
        d_vectorA, d_vectorB, kVectorSize, kScale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(vectorStop));
    CUDA_CHECK(cudaEventSynchronize(vectorStop));

    float vectorGpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&vectorGpuTimeMs, vectorStart, vectorStop));
    CUDA_CHECK(cudaMemcpy(h_vectorGpu.data(),
                          d_vectorB,
                          vectorBytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(vectorStart));
    CUDA_CHECK(cudaEventDestroy(vectorStop));
    CUDA_CHECK(cudaFree(d_vectorA));
    CUDA_CHECK(cudaFree(d_vectorB));

    const bool vectorOk = compareVectors(h_vectorCpu, h_vectorGpu);

    std::vector<unsigned char> h_imageInput(kImageWidth * kImageHeight);
    std::vector<unsigned char> h_imageCpu(kImageWidth * kImageHeight);
    std::vector<unsigned char> h_imageGpu(kImageWidth * kImageHeight);

    for (unsigned char& pixel : h_imageInput) {
        pixel = static_cast<unsigned char>(byteDistribution(generator));
    }

    const double imageCpuTimeMs = measureCpuMilliseconds([&]() {
        thresholdFilterCPU(h_imageInput.data(),
                           h_imageCpu.data(),
                           kImageWidth,
                           kImageHeight,
                           kThreshold);
    });

    unsigned char* d_imageInput = nullptr;
    unsigned char* d_imageOutput = nullptr;
    const size_t imageBytes = h_imageInput.size() * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_imageInput), imageBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_imageOutput), imageBytes));
    CUDA_CHECK(cudaMemcpy(d_imageInput,
                          h_imageInput.data(),
                          imageBytes,
                          cudaMemcpyHostToDevice));

    const dim3 imageThreadsPerBlock(16, 16);
    const dim3 imageBlocks((kImageWidth + imageThreadsPerBlock.x - 1) /
                               imageThreadsPerBlock.x,
                           (kImageHeight + imageThreadsPerBlock.y - 1) /
                               imageThreadsPerBlock.y);

    cudaEvent_t imageStart = nullptr;
    cudaEvent_t imageStop = nullptr;
    CUDA_CHECK(cudaEventCreate(&imageStart));
    CUDA_CHECK(cudaEventCreate(&imageStop));
    CUDA_CHECK(cudaEventRecord(imageStart));
    thresholdFilterCUDA<<<imageBlocks, imageThreadsPerBlock>>>(
        d_imageInput, d_imageOutput, kImageWidth, kImageHeight, kThreshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(imageStop));
    CUDA_CHECK(cudaEventSynchronize(imageStop));

    float imageGpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&imageGpuTimeMs, imageStart, imageStop));
    CUDA_CHECK(cudaMemcpy(h_imageGpu.data(),
                          d_imageOutput,
                          imageBytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(imageStart));
    CUDA_CHECK(cudaEventDestroy(imageStop));
    CUDA_CHECK(cudaFree(d_imageInput));
    CUDA_CHECK(cudaFree(d_imageOutput));

    const bool imageOk = compareImages(h_imageCpu, h_imageGpu);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Task 1: vector scaling\n";
    std::cout << "N = " << kVectorSize << ", k = " << kScale << '\n';
    std::cout << "CPU time: " << vectorCpuTimeMs << " ms\n";
    std::cout << "GPU kernel time: " << vectorGpuTimeMs << " ms\n";
    std::cout << "Result: " << (vectorOk ? "OK" : "FAILED") << "\n\n";

    std::cout << "Task 2: binary thresholding\n";
    std::cout << "Image = " << kImageWidth << "x" << kImageHeight
              << ", T = " << static_cast<int>(kThreshold) << '\n';
    std::cout << "CPU time: " << imageCpuTimeMs << " ms\n";
    std::cout << "GPU kernel time: " << imageGpuTimeMs << " ms\n";
    std::cout << "Result: " << (imageOk ? "OK" : "FAILED") << '\n';

    CUDA_CHECK(cudaDeviceReset());

    return vectorOk && imageOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
