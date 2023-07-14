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

template <typename TElement>
void VAIExecutionProviderTest(TElement indices[]) {

  // TODO: initialize execution provider and get device from it
  // Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
  // ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

  /* or another option is via command queue:
  * ComPtr<ID3D12Device> device;
  GRAPHICS_THROW_IF_FAILED(dxqueue->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));*/

  // Alternativelly create the d3d device

  // TODO: dodja ovde sa adapterom create device

  /*ComPtr<IDXGIFactory4> dxgi_factory;
  ORT_THROW_IF_FAILED(CreateDXGIFactory2(0, IID_GRAPHICS_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())));
  ComPtr<IDXGIAdapter1> adapter;
  ORT_THROW_IF_FAILED(dxgi_factory->EnumAdapters1(device_id, &adapter));
  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));*/

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

  size_t byteSize = sizeof(indices) * sizeof(TElement);

  ComPtr<ID3D12Resource> bufferUploader = nullptr;
  ComPtr<ID3D12Resource> bufferGPU = nullptr;

  // copy CPU buffer to GPU
  bufferGPU = vaip::tensor_proto_new_d3d12_cpu_to_gpu(d3d12_device.Get(), bufferUploader, cmdList.Get(),
                                                indices, byteSize);

  // copy GPU buffer to CPU (readback)  void* redbackCpuResult = 
  void* dst = nullptr;
  
  dst = vaip::tensor_proto_new_d3d12_gpu_to_cpu(bufferGPU, d3d12_device.Get(), cmdList.Get(),
                                                             byteSize, dxQueue);
  //// TODO: Write to file
  //std::ofstream fw("result1.txt", std::ofstream::out);

  //// check if file was successfully opened for writing
  //if (fw.is_open()) {
  //  // store array contents to text file
  //  for (int i = 0; i < 18; ++i) {
  //    fw << static_cast<uint16_t*>(dst)[i] << "\n";
  //  }
  //  fw.close();
  //} else
  //  std::cout << "Problem with opening file";
}

// add command to test function written above, need to fix include issues to add it
TEST(VitisAIExecutionProviderTest, VAIPTest) {
  //add input data here
  /*uint16_t indices[] = {
      1, 1, 2,
      5, 2, 3,
      4, 6, 5,
      4, 7, 6,
      4, 5, 1,
      4, 1, 0};*/
  uint16_t indices[] = {1};

  VAIExecutionProviderTest<uint16_t>(indices);
}

}  // namespace test
}  // namespace onnxruntime