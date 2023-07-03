#include <wrl.h>
#include "C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um\d3d12.h"
#include <cstddef>
#include <stdint.h>
#include "C:\Users\tvukovic\dev\onnxruntime\onnxruntime\core\providers\dml\DmlExecutionProvider\src\ErrorHandling.h"
#include "C:\Users\tvukovic\dev\onnxruntime\onnxruntime\core\providers\vitisai\imp\tensor_proto.h"
//#include "C:\Users\tvukovic\dev\onnxruntime\onnxruntime\core\framework\error_code_helper.h"

//#include "C:\Users\tvukovic\dev\onnxruntime\build\Windows\Release\_deps\googletest-src\googletest\include\gtest\gtest.h"

namespace onnxruntime {
namespace test {

using namespace vaip;
using Microsoft::WRL::ComPtr;

template <typename TElement>
void VAIPTest(TElement indices[]) {

  // TODO: initialize execution provider and get device from it
  // Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
  // ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

  /* or another option is via command queue:
  * ComPtr<ID3D12Device> device;
  GRAPHICS_THROW_IF_FAILED(dxqueue->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));*/

  // Alternativelly create the d3d device.ORT_THROW_IF_FAILED
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

  TElement byteSize = sizeof(indices) * sizeof(TElement);

  ComPtr<ID3D12Resource> bufferUploader = nullptr;
  ComPtr<ID3D12Resource> bufferGPU = nullptr;

  // copy CPU buffer to GPU
  bufferGPU = vaip::tensor_proto_new_d3d12_cpu_to_gpu(d3d12_device.Get(), bufferUploader, cmdList.Get(),
                                                indices, byteSize);

  // copy GPU buffer to CPU (readback)
  void* redbackCpuResult = vaip::tensor_proto_new_d3d12_gpu_to_cpu(bufferGPU, d3d12_device.Get(), cmdList.Get(),
                                                             byteSize, dxQueue);

}

// add command to test function written above, need to fix include issues to add it
//TEST(vai_basic_test, VAIPTest) {
//  // add input data here
//  uint16_t indices[] = {
//      0, 1, 2,
//      0, 2, 3,
//      4, 6, 5,
//      4, 7, 6,
//      4, 5, 1,
//      4, 1, 0};
//
//  VAIPTest<uint16_t>(indices);
//}

int main() {
    // add input data here
    uint16_t indices[] = {
        0, 1, 2,
        0, 2, 3,
        4, 6, 5,
        4, 7, 6,
        4, 5, 1,
        4, 1, 0};
  
    VAIPTest<uint16_t>(indices);
    return 0;
}

}  // namespace test
}  // namespace onnxruntime