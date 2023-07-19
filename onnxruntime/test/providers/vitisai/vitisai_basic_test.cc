#pragma comment(lib, "d3d12.lib")

#include "core/providers/vitisai/imp/tensor_proto.h"

#include <wil/result.h>
#include <gtest/gtest.h>
#include <cstddef>

#include <iostream>
#include <string>
#include <fstream>

using namespace vaip;
using Microsoft::WRL::ComPtr;

namespace onnxruntime {
namespace test {

LARGE_INTEGER getStartingTime() {
  LARGE_INTEGER startingTime;
  QueryPerformanceCounter(&startingTime);
  return startingTime;
}

int getElapsedTime(LARGE_INTEGER startingTime) {
  LARGE_INTEGER endingTime, elapsedMicroseconds, frequency;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&endingTime);
  elapsedMicroseconds.QuadPart = endingTime.QuadPart - startingTime.QuadPart;
  //
  // We now have the elapsed number of ticks, along with the
  // number of ticks-per-second. We use these values
  // to convert to the number of elapsed microseconds.
  // To guard against loss-of-precision, we convert
  // to microseconds *before* dividing by ticks-per-second.
  //
  elapsedMicroseconds.QuadPart *= 1000000;
  elapsedMicroseconds.QuadPart /= frequency.QuadPart;

  return elapsedMicroseconds.QuadPart;
}

struct uploadReadback {
  float upload, readback;
};

typedef struct uploadReadback Struct;

template <typename TElement>
Struct VAIExecutionProviderTest(std::vector<TElement> indices, int size) {

  ComPtr<ID3D12Device> d3d12_device; 
  
  ORT_THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  ComPtr<ID3D12CommandQueue> dxQueue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> cmdList;

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

  // create command queue 
  ORT_THROW_IF_FAILED(d3d12_device->CreateCommandQueue(
      &queueDesc, IID_PPV_ARGS(&dxQueue)));

  // create command allocator 
  ORT_THROW_IF_FAILED(d3d12_device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT,
      IID_PPV_ARGS(allocator.GetAddressOf())));

  // create command list 
  ORT_THROW_IF_FAILED(d3d12_device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_DIRECT,
      allocator.Get(),  // Associated command allocator
      nullptr,          // Initial PipelineStateObject
      IID_PPV_ARGS(cmdList.GetAddressOf())));

  size_t byteSize = size * sizeof(TElement);

  ComPtr<ID3D12Resource> bufferUploader = nullptr;
  ComPtr<ID3D12Resource> bufferGPU = nullptr;

  // calling CPU->GPU function and measuring time
  LARGE_INTEGER start = getStartingTime();
  // copy CPU buffer to GPU
  bufferGPU = vaip::tensor_proto_new_d3d12_cpu_to_gpu(d3d12_device.Get(), bufferUploader, cmdList.Get(),
                                                      static_cast<void*>(indices.data()), byteSize);
  int uploadElapsed = getElapsedTime(start);

  // copy GPU buffer to CPU (readback)  
  void* output = nullptr;

  // calling GPU->CPU readback function and measuring time
  start = getStartingTime();
  // copy GPU buffer to CPU (readback)
  output = vaip::tensor_proto_new_d3d12_gpu_to_cpu(bufferGPU, d3d12_device.Get(), cmdList.Get(),
                                                   byteSize, dxQueue);
  
  int readbackElapsed = getElapsedTime(start);

  //// Write output to file
  // std::ofstream fw("result_dst.txt", std::ofstream::out);
  //// check if file was successfully opened for writing
  // if (fw.is_open()) {
  //   // store array contents to text file
  //   for (int i = 0; i < size; i++) {
  //     fw << static_cast<int*>(output)[i] << "\n";
  //   }
  //   fw.close();
  // }
  free(output);  

  Struct s;
  s.upload = uploadElapsed;
  s.readback = readbackElapsed;
  return s;
}

TEST(VitisAIExecutionProviderTest, VAIPTest) {
  
  // adding input data
  std::vector<int8_t> int8_10k(10000);
  std::vector<int8_t> int8_100k(100000);
  std::vector<int8_t> int8_1m(1000000);
  std::vector<int8_t> int8_10m(10000000);

  // check if float is 32 bit
  assert(CHAR_BIT * sizeof(float) == 32);

  // create fp32 input data
  std::vector<float> fp32_10k(10000); 
  std::vector<float> fp32_100k(100000);
  std::vector<float> fp32_1m(1000000);
  std::vector<float> fp32_10m(10000000);

  Struct res_fp32_10k = VAIExecutionProviderTest<float>(fp32_10k, 10000);
  Struct res_int8_10k = VAIExecutionProviderTest<int8_t>(int8_10k, 10000);
  Struct res_fp32_100k = VAIExecutionProviderTest<float>(fp32_100k, 100000);
  Struct res_int8_100k = VAIExecutionProviderTest<int8_t>(int8_100k, 100000);
  Struct res_fp32_1m = VAIExecutionProviderTest<float>(fp32_1m, 1000000);
  Struct res_int8_1m = VAIExecutionProviderTest<int8_t>(int8_1m, 1000000);
  Struct res_fp32_10m = VAIExecutionProviderTest<float>(fp32_10m, 10000000);
  Struct res_int8_10m = VAIExecutionProviderTest<int8_t>(int8_10m, 10000000);

  std::ofstream myfile;
  myfile.open("results.csv");
  myfile << "tensor size, int8 (GPU->CPU) [us], int8 (CPU->GPU) [us],fp32 (GPU->CPU) [us], fp32 (CPU->GPU) [us],\n";
  myfile << "10k," << res_int8_10k.readback << "," << res_int8_10k.upload << "," << res_fp32_10k.readback << "," << res_fp32_10k.upload
         << ",\n ";
  myfile << "100k," << res_int8_100k.readback << "," << res_int8_100k.upload << "," << res_fp32_100k.readback << "," << res_fp32_100k.upload
         << ",\n ";
  myfile << "1m," << res_int8_1m.readback << "," << res_int8_1m.upload << "," << res_fp32_1m.readback << "," << res_fp32_1m.upload
         << ",\n ";
  myfile << "10m," << res_int8_10m.readback << "," << res_int8_10m.upload << "," << res_fp32_10m.readback << "," << res_fp32_10m.upload
         << ",\n ";
  myfile.close();
}

}  // namespace test
}  // namespace onnxruntime