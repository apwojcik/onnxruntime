#pragma comment(lib, "d3d12.lib")

#include "core/providers/vitisai/imp/tensor_proto.h"

#include <wil/result.h>
#include <gtest/gtest.h>
#include <cstddef>

#include <iostream>
#include <string>
#include <fstream>

using namespace vaip;
using namespace std;
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
Struct VAIExecutionProviderTest(TElement* indices, int x, int y, int z) { 

  ComPtr<ID3D12Device> d3d12_device; 

  int size = x * y * z;
  
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
                                                      static_cast<void*>(indices), byteSize);
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
  //   for (int i = 0; i < x; i++) {
  //     for (int j = 0; j < y; j++) {
  //       for (int k = 0; k < z; k++) {
  //         fw << "A[" << i << "][" << j << "][" << k << "] " << static_cast<int*>(output)[i + x * (j + y * k)];
  //         fw << "\n";
  //       }
  //      
  //     }
  //   }
  //   fw.close();
  // }
  free(output);  

  Struct s;
  s.upload = uploadElapsed;
  s.readback = readbackElapsed;
  return s;
}

template <typename TElement>
Struct VAIExecutionProviderTest(TElement* indices, int size) {  

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
                                                      static_cast<void*>(indices), byteSize);
  
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
  // std::ofstream fw("result_dst1d.txt", std::ofstream::out);
  //// check if file was successfully opened for writing
  // if (fw.is_open()) {
  //   // store array contents to text file
  //   for (int i = 0; i < size; i++) {
  //         fw << static_cast<int*>(output)[i];
  //         fw << "\n";
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

  // add input rgb tensor wxhx4 int8
  int8_t* array1 = new int8_t[3840 * 1260 * 4];
  int8_t* array2 = new int8_t[2560 * 1440 * 4];
  int8_t* array3 = new int8_t[1920 * 1080 * 4];

  Struct res_3840x1260 = VAIExecutionProviderTest<int8_t>(array1, 3840, 1260, 4);
  delete[] array1;
  Struct res_2560x1440 = VAIExecutionProviderTest<int8_t>(array2, 2560, 1440, 4);
  delete[] array2;
  Struct res_1920x1080 = VAIExecutionProviderTest<int8_t>(array3, 1920, 1080, 4);
  delete[] array3;

  // 1D input data
  int8_t* int8_10k = new int8_t[10000];
  int8_t* int8_100k = new int8_t[100000];
  int8_t* int8_1m = new int8_t[1000000];
  int8_t* int8_10m = new int8_t[10000000];

  // check if float is 32 bit
  assert(CHAR_BIT * sizeof(float) == 32);

  float* fp32_10k = new float[10000];
  float* fp32_100k = new float[100000];
  float* fp32_1m = new float[1000000];
  float* fp32_10m = new float[10000000];

   Struct res_fp32_10k = VAIExecutionProviderTest<float>(fp32_10k, 10000);
   Struct res_int8_10k = VAIExecutionProviderTest<int8_t>(int8_10k, 10000);
   Struct res_fp32_100k = VAIExecutionProviderTest<float>(fp32_100k, 100000);
   Struct res_int8_100k = VAIExecutionProviderTest<int8_t>(int8_100k, 100000);
   Struct res_fp32_1m = VAIExecutionProviderTest<float>(fp32_1m, 1000000);
   Struct res_int8_1m = VAIExecutionProviderTest<int8_t>(int8_1m, 1000000);
   Struct res_fp32_10m = VAIExecutionProviderTest<float>(fp32_10m, 10000000);
   Struct res_int8_10m = VAIExecutionProviderTest<int8_t>(int8_10m, 10000000);

   delete[] int8_10k;
   delete[] int8_100k;
   delete[] int8_1m;
   delete[] int8_10m;
   delete[] fp32_10k;
   delete[] fp32_100k;
   delete[] fp32_1m;
   delete[] fp32_10m;

  system("wmic path win32_VideoController get name > gpu_info.txt");
  system("wmic cpu get name > sys_info.txt");
  system("wmic memorychip get speed >> sys_info.txt");

  std::ifstream file("results.txt");
  std::string str;
  std::string GPUName;
  std::string CPUName;
  std::string memorySpeed;

  int count = 0;
  while (std::getline(file, str)) {
   if (count == 1) {
      CPUName = str;
    } else if (count == 3) {
      memorySpeed = str;
    }
    count++;
  }  

  std::ifstream file_gpu("gpu_info.txt");
  count = 0;
  while (std::getline(file_gpu, str)) {
    if (count == 2) {
      GPUName = str;
      break;
    } 
    count++;
  }  

  std::ofstream myfile;
  myfile.open("sys_info.csv");
  
  myfile << "System information: \n";
  myfile << "CPU:," << CPUName;
  myfile << "GPU:," << GPUName;
  myfile << "Memory speed [MHz]:," << memorySpeed;
  myfile << "Bus speed [MHz]:,"
         << "99.98 \n";
  myfile << "\n";

  myfile << "2D Performance (RGBA): \n";
  myfile << "tensor size,GPU->CPU [us], CPU->GPU [us] \n";
  myfile << "1920x1080 ~8M," << res_1920x1080.readback << "," << res_1920x1080.upload << ",\n ";
  myfile << "2560x1440 ~15M," << res_2560x1440.readback << "," << res_2560x1440.upload << ",\n ";
  myfile << "3840x1260 ~19M," << res_3840x1260.readback << "," << res_3840x1260.upload << ",\n ";
  myfile << "\n";

  myfile << "1D Performance: \n";
  myfile << "tensor size, int8 (GPU->CPU) [us], int8 (CPU->GPU) [us],fp32 (GPU->CPU) [us], fp32 (CPU->GPU) [us] \n";
  myfile << "10K," << res_int8_10k.readback << "," << res_int8_10k.upload << "," << res_fp32_10k.readback << "," << res_fp32_10k.upload
         << ",\n ";
  myfile << "100K," << res_int8_100k.readback << "," << res_int8_100k.upload << "," << res_fp32_100k.readback << "," << res_fp32_100k.upload
         << ",\n ";
  myfile << "1M," << res_int8_1m.readback << "," << res_int8_1m.upload << "," << res_fp32_1m.readback << "," << res_fp32_1m.upload
         << ",\n ";
  myfile << "10M," << res_int8_10m.readback << "," << res_int8_10m.upload << "," << res_fp32_10m.readback << "," << res_fp32_10m.upload
         << ",\n ";
  myfile.close();
}

}  // namespace test
}  // namespace onnxruntime