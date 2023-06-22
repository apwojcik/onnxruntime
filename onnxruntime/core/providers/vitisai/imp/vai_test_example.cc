#include <wrl.h>
#include "C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um\d3d12.h"
#include <cstddef>
#include "C:\Users\tvukovic\dev\onnxruntime\onnxruntime\core\providers\dml\DmlExecutionProvider\src\ErrorHandling.h"

using Microsoft::WRL::ComPtr;

ComPtr<ID3D12Resource>
     CreateCpuResource(
         const std::byte* tensorPtr,
         size_t tensorByteSize) {
       ComPtr<ID3D12Resource> buffer;
    
       D3D12_HEAP_PROPERTIES heapProperties = {
           D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0};
    
       D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                           0,
                                           static_cast<u_int64>((tensorByteSize + 3) & ~3),
                                           1,
                                           1,
                                           1,
                                           DXGI_FORMAT_UNKNOWN,
                                           {1, 0},
                                           D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    
       ComPtr<ID3D12Device> d3dDevice;
       /*ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));*/
    
       d3dDevice->CreateCommittedResource(
           &heapProperties,
           D3D12_HEAP_FLAG_NONE,
           &resourceDesc,
           D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
           nullptr,
           IID_PPV_ARGS(&buffer)); // ovde dodati ORT_THROW_IF_FAILED
    
       // Map the buffer and copy the data
       void* bufferData = nullptr;
       D3D12_RANGE range = {0, tensorByteSize};
       buffer->Map(0, &range, &bufferData); // ovde dodati ORT_THROW_IF_FAILED
       memcpy(bufferData, tensorPtr, tensorByteSize);
       buffer->Unmap(0, &range);
    
       return buffer;
     }

int main() {
       ComPtr<ID3D12Resource> buffer = CreateCpuResource(nullptr, 64 * 1024);
    return 0;
}