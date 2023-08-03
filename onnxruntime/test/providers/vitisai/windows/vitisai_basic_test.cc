#pragma comment(lib, "d3d12.lib")

#include "core/providers/vitisai/imp/tensor_proto.h"

#include <wil/result.h>
#include <gtest/gtest.h>
#include <cstddef>

#include <iostream>
#include <fstream>

namespace onnxruntime {
namespace test {

struct elapsedTime {
  float upload, readback, memcpyElapsedTime,readbackBufferElapsedTime,GPUResourceElapsedTime,UploadBufferElapsedTime;
};

struct shapeRGBA {
  int width, height, bytes;
};

template <typename TElement>
elapsedTime VAIExecutionProviderTest(std::unique_ptr<TElement[]> inputs, int x, int y, int z) {
  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;

  int size = x * y * z;

  ORT_THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  Microsoft::WRL::ComPtr<ID3D12CommandQueue> dxQueue;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList;

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

  Microsoft::WRL::ComPtr<ID3D12Resource> bufferUploader = nullptr;
  Microsoft::WRL::ComPtr<ID3D12Resource> bufferGPU = nullptr;

  int memcpyElapsedTime, readbackBufferElapsedTime, GPUResourceElapsedTime, UploadBufferElapsedTime;

  // calling CPU->GPU function and measuring time
  LARGE_INTEGER start = vaip::getStartingTime();
  // copy CPU buffer to GPU
  std::tie(bufferGPU, GPUResourceElapsedTime, UploadBufferElapsedTime) = vaip::tensor_proto_new_d3d12_cpu_to_gpu(d3d12_device.Get(), bufferUploader, cmdList.Get(),
                                                      inputs.release(), byteSize);
  int uploadElapsed = vaip::getElapsedTime(start);

  // copy GPU buffer to CPU (readback)
  void* output = nullptr;

  // calling GPU->CPU readback function and measuring time
  start = vaip::getStartingTime();
  // copy GPU buffer to CPU (readback)
  std::tie(output, memcpyElapsedTime, readbackBufferElapsedTime) = vaip::tensor_proto_new_d3d12_gpu_to_cpu(bufferGPU, d3d12_device.Get(), cmdList.Get(),
                                                   byteSize, dxQueue);

  int readbackElapsed = vaip::getElapsedTime(start);

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
  // }
  if (output != NULL) free(output);


  elapsedTime s;
  s.upload = uploadElapsed;
  s.readback = readbackElapsed;
  s.memcpyElapsedTime = memcpyElapsedTime;
  s.readbackBufferElapsedTime = readbackBufferElapsedTime;
  s.GPUResourceElapsedTime = GPUResourceElapsedTime;
  s.UploadBufferElapsedTime = UploadBufferElapsedTime;
  return s;
}

template <typename TElement>
elapsedTime VAIExecutionProviderTest(std::unique_ptr<TElement[]> inputs, int size) {
  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;

  ORT_THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

  Microsoft::WRL::ComPtr<ID3D12CommandQueue> dxQueue;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList;

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

  Microsoft::WRL::ComPtr<ID3D12Resource> bufferUploader = nullptr;
  Microsoft::WRL::ComPtr<ID3D12Resource> bufferGPU = nullptr;

  int memcpyElapsedTime, readbackBufferElapsedTime, GPUResourceElapsedTime, UploadBufferElapsedTime;

  // calling CPU->GPU function and measuring time
  LARGE_INTEGER start = vaip::getStartingTime();
  // copy CPU buffer to GPU
  std::tie(bufferGPU, GPUResourceElapsedTime, UploadBufferElapsedTime) = vaip::tensor_proto_new_d3d12_cpu_to_gpu(d3d12_device.Get(), bufferUploader, cmdList.Get(),
                                                      inputs.release(), byteSize);

  int uploadElapsed = vaip::getElapsedTime(start);

  // copy GPU buffer to CPU (readback)
  void* output = nullptr;

  // calling GPU->CPU readback function and measuring time
  start = vaip::getStartingTime();
  // copy GPU buffer to CPU (readback)
  std::tie(output, memcpyElapsedTime, readbackBufferElapsedTime) = vaip::tensor_proto_new_d3d12_gpu_to_cpu(bufferGPU, d3d12_device.Get(), cmdList.Get(),
                                                   byteSize, dxQueue);

  int readbackElapsed = vaip::getElapsedTime(start);

  //// Write output to file
  // std::ofstream fw("result_dst1d.txt", std::ofstream::out);
  //// check if file was successfully opened for writing
  // if (fw.is_open()) {
  //   // store array contents to text file
  //   for (int i = 0; i < size; i++) {
  //         fw << static_cast<int*>(output)[i];
  //         fw << "\n";
  //   }
  // }

  if (output != NULL) free(output);

  elapsedTime s;
  s.upload = uploadElapsed;
  s.readback = readbackElapsed;
  s.memcpyElapsedTime = memcpyElapsedTime;
  s.readbackBufferElapsedTime = readbackBufferElapsedTime;
  s.GPUResourceElapsedTime = GPUResourceElapsedTime;
  s.UploadBufferElapsedTime = UploadBufferElapsedTime;
  return s;
}

TEST(VitisAIExecutionProviderTest, VAIPTest) {

  shapeRGBA shapeRGBA1 = {3840, 2160, 4};
  shapeRGBA shapeRGBA2 = {2560, 1440, 4};
  shapeRGBA shapeRGBA3 = {1920, 1080, 4};
  // add input RGBA int8 tensor with shape wxhx4 
  auto rgba1 = std::make_unique<int8_t[]>(shapeRGBA1.width * shapeRGBA1.height * shapeRGBA1.bytes);
  auto rgba2 = std::make_unique<int8_t[]>(shapeRGBA2.width * shapeRGBA2.height * shapeRGBA2.bytes);
  auto rgba3 = std::make_unique<int8_t[]>(shapeRGBA3.width * shapeRGBA3.height * shapeRGBA3.bytes);
  elapsedTime res_3840x2160 = VAIExecutionProviderTest<int8_t>(std::move(rgba1), shapeRGBA1.width, shapeRGBA1.height, shapeRGBA1.bytes);
  elapsedTime res_2560x1440 = VAIExecutionProviderTest<int8_t>(std::move(rgba2), shapeRGBA2.width, shapeRGBA2.height, shapeRGBA2.bytes);
  elapsedTime res_1920x1080 = VAIExecutionProviderTest<int8_t>(std::move(rgba3), shapeRGBA3.width, shapeRGBA3.height, shapeRGBA3.bytes);

  // 1D input data
  int size1D10k = 10000;
  int size1D100k = 100000;
  int size1D1m = 1000000;
  int size1D10m = 10000000;

  auto int8_10k = std::make_unique<int8_t[]>(size1D10k);
  auto int8_100k = std::make_unique<int8_t[]>(size1D100k);
  auto int8_1m = std::make_unique<int8_t[]>(size1D1m);
  auto int8_10m = std::make_unique<int8_t[]>(size1D10m);

  auto fp32_10k = std::make_unique<float[]>(size1D10k);
  auto fp32_100k = std::make_unique<float[]>(size1D100k);
  auto fp32_1m = std::make_unique<float[]>(size1D1m);
  auto fp32_10m = std::make_unique<float[]>(size1D10m);

  elapsedTime res_fp32_10k = VAIExecutionProviderTest<float>(std::move(fp32_10k), size1D10k);
  elapsedTime res_int8_10k = VAIExecutionProviderTest<int8_t>(std::move(int8_10k), size1D10k);
  elapsedTime res_fp32_100k = VAIExecutionProviderTest<float>(std::move(fp32_100k), size1D100k);
  elapsedTime res_int8_100k = VAIExecutionProviderTest<int8_t>(std::move(int8_100k), size1D100k);
  elapsedTime res_fp32_1m = VAIExecutionProviderTest<float>(std::move(fp32_1m), size1D1m);
  elapsedTime res_int8_1m = VAIExecutionProviderTest<int8_t>(std::move(int8_1m), size1D1m);
  elapsedTime res_fp32_10m = VAIExecutionProviderTest<float>(std::move(fp32_10m), size1D10m);
  elapsedTime res_int8_10m = VAIExecutionProviderTest<int8_t>(std::move(int8_10m), size1D10m);

  std::system("wmic path win32_VideoController get name > gpu_info.txt");
  std::system("wmic cpu get name > sys_info.txt");
  std::system("wmic memorychip get speed >> sys_info.txt");

  std::ifstream file("sys_info.txt");
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

  std::ofstream myfile{"results.csv"};

  myfile << "System information: \n";
  myfile << "CPU:," << CPUName;
  myfile << "GPU:," << GPUName;
  myfile << "Memory speed [MHz]:," << memorySpeed;
  myfile << "Bus speed [MHz]:,"
         << "99.98 \n";
  myfile << "\n";

  myfile << "2D Performance (RGBA): \n";
  myfile << "tensor size,GPU->CPU [us], GPU->CPU memcpy [us], GPU->CPU readbackBufferElapsedTime [us],CPU->GPU [us], CPU->GPU GPUResourceElapsedTime [us],  CPU->GPU UploadBufferElapsedTime [us]  \n";
  myfile << "1920x1080 ~8M," << res_1920x1080.readback << "," << res_1920x1080.memcpyElapsedTime << "," << res_1920x1080.readbackBufferElapsedTime << "," << res_1920x1080.upload << "," << res_1920x1080.GPUResourceElapsedTime << "," << res_1920x1080.UploadBufferElapsedTime << ",\n ";
  myfile << "2560x1440 ~15M," << res_2560x1440.readback << "," << res_2560x1440.memcpyElapsedTime << "," << res_2560x1440.readbackBufferElapsedTime << "," << res_2560x1440.upload << "," << res_2560x1440.GPUResourceElapsedTime << "," << res_2560x1440.UploadBufferElapsedTime << ",\n ";
  myfile << "3840x2160 ~33M," << res_3840x2160.readback << "," << res_3840x2160.memcpyElapsedTime << "," << res_3840x2160.readbackBufferElapsedTime << "," << res_3840x2160.upload << "," << res_3840x2160.GPUResourceElapsedTime << "," << res_3840x2160.UploadBufferElapsedTime << ",\n ";
  myfile << "\n";

  myfile << "1D Performance: \n";
  myfile << "tensor size, int8 (GPU->CPU) [us], int8 (GPU->CPU memcpy) [us], int8 (GPU->CPU readbackBufferElapsedTime) [us], int8 (CPU->GPU) [us], int8 (CPU->GPU GPUResourceElapsedTime) [us],  int8 (CPU->GPU UploadBufferElapsedTime) [us],fp32 (GPU->CPU) [us], fp32 (GPU->CPU memcpy) [us], fp32 (GPU->CPU readbackBufferElapsedTime) [us],fp32 (CPU->GPU) [us], fp32 (CPU->GPU GPUResourceElapsedTime) [us],  fp32 (CPU->GPU UploadBufferElapsedTime) [us] \n";
  myfile << "10K," << res_int8_10k.readback << "," << res_int8_10k.memcpyElapsedTime << "," << res_int8_10k.readbackBufferElapsedTime << "," << res_int8_10k.upload << "," << res_int8_10k.GPUResourceElapsedTime << "," << res_int8_10k.UploadBufferElapsedTime << "," << res_fp32_10k.readback << "," << res_fp32_10k.memcpyElapsedTime << "," << res_fp32_10k.readbackBufferElapsedTime << "," << res_fp32_10k.upload
         << "," << res_fp32_10k.GPUResourceElapsedTime << "," << res_fp32_10k.UploadBufferElapsedTime << ",\n ";
  myfile << "100K," << res_int8_100k.readback << "," << res_int8_100k.memcpyElapsedTime << "," << res_int8_100k.readbackBufferElapsedTime << "," << res_int8_100k.upload << "," << res_int8_100k.GPUResourceElapsedTime << "," << res_int8_100k.UploadBufferElapsedTime << "," << res_fp32_100k.readback << "," << res_fp32_100k.memcpyElapsedTime << "," << res_fp32_100k.readbackBufferElapsedTime << "," << res_fp32_100k.upload
         << "," << res_fp32_100k.GPUResourceElapsedTime << "," << res_fp32_100k.UploadBufferElapsedTime << ",\n ";
  myfile << "1M," << res_int8_1m.readback << "," << res_int8_1m.memcpyElapsedTime << "," << res_int8_1m.readbackBufferElapsedTime << "," << res_int8_1m.upload << "," << res_int8_1m.GPUResourceElapsedTime << "," << res_int8_1m.UploadBufferElapsedTime << "," << res_fp32_1m.readback << "," << res_fp32_1m.memcpyElapsedTime << "," << res_fp32_1m.readbackBufferElapsedTime << "," << res_fp32_1m.upload
         << "," << res_fp32_1m.GPUResourceElapsedTime << "," << res_fp32_1m.UploadBufferElapsedTime << ",\n ";
  myfile << "10M," << res_int8_10m.readback << "," << res_int8_10m.memcpyElapsedTime << "," << res_int8_10m.readbackBufferElapsedTime << "," << res_int8_10m.upload << "," << res_int8_10m.GPUResourceElapsedTime << "," << res_int8_10m.UploadBufferElapsedTime << "," << res_fp32_10m.readback << "," << res_fp32_10m.memcpyElapsedTime << "," << res_fp32_10m.readbackBufferElapsedTime << "," << res_fp32_10m.upload
         << "," << res_fp32_10m.GPUResourceElapsedTime << "," << res_fp32_10m.UploadBufferElapsedTime << ",\n ";
}
}  // namespace test
}  // namespace onnxruntime