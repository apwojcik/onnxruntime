// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./tensor_proto.h"
#include "./vai_assert.h"

#include <iostream>
#include <string>
#include <fstream>


namespace vaip {
using Microsoft::WRL::ComPtr;

void FlushCommandQueue(ComPtr<ID3D12CommandQueue> cmdQueue, ID3D12Device* device) {
    // CPU GPU synchronization
    // flushing cmd queue using a fence
    //
    // Create Event
    HANDLE directEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    wil::unique_event hDirectEvent(directEvent);

    // Create Fence
    ComPtr<ID3D12Fence> spDirectFence = nullptr;
    ORT_THROW_IF_FAILED(device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(spDirectFence.ReleaseAndGetAddressOf())));

    // Adds fence to queue
    ORT_THROW_IF_FAILED(cmdQueue->Signal(spDirectFence.Get(), FENCE_SIGNAL_VALUE));
    ORT_THROW_IF_FAILED(spDirectFence->SetEventOnCompletion(FENCE_SIGNAL_VALUE, hDirectEvent.get()));

    // Wait for signal
    DWORD retVal = WaitForSingleObject(hDirectEvent.get(), INFINITE);
    if (retVal != WAIT_OBJECT_0) {
      ORT_THROW_IF_FAILED(E_UNEXPECTED);
    }
}

gsl::span<const char> tensor_proto_as_raw(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  auto data_type = tensor.data_type();
  auto& mut_tensor = const_cast<ONNX_NAMESPACE::TensorProto&>(tensor);
  if (tensor.has_raw_data()) {
    return gsl::span<const char>(tensor.raw_data().data(), tensor.raw_data().size());
  } else if (tensor.float_data_size() > 0 && data_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
    return gsl::span<const char>((char*)tensor.float_data().data(), tensor.float_data().size() * sizeof(float));
  } else if (tensor.int32_data_size() > 0 && data_type == ONNX_NAMESPACE::TensorProto::INT32) {
    return gsl::span<const char>((char*)tensor.int32_data().data(), tensor.int32_data().size() * sizeof(int));
    // test case: graph_opt model #43
  } else if (tensor.int64_data_size() > 0 && data_type == ONNX_NAMESPACE::TensorProto::INT64) {
    return gsl::span<const char>((char*)tensor.int64_data().data(), tensor.int64_data().size() * sizeof(int64_t));
  } else if (data_type == ONNX_NAMESPACE::TensorProto::INT8) {
    auto size = tensor.int32_data_size();
    assert(size > 0);
    mut_tensor.mutable_raw_data()->resize(sizeof(char) * size);
    char* base = &(*mut_tensor.mutable_raw_data())[0];
    for (auto i = 0; i < size; ++i) {
      auto value = (char)tensor.int32_data(i);
      assert(value >= std::numeric_limits<char>::min());
      assert(value <= std::numeric_limits<char>::max());
      base[i] = value;
    }
    return gsl::span<const char>(tensor.raw_data().data(), tensor.raw_data().size());
  } else {
    vai_assert(false, "not support data_type");
  }
#ifndef _WIN32
  return gsl::span<const char>(tensor.raw_data().data(), tensor.raw_data().size());
#endif
  return gsl::span<const char>();
}

size_t tensor_proto_raw_data_size(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.raw_data().size();
}

std::vector<int64_t> tensor_proto_get_shape(
    const onnx::TensorProto& tensor_proto) {
  auto ret = std::vector<int64_t>();
  int rank = tensor_proto.dims_size();
  if (rank > 0) {
    ret.reserve((size_t)rank);
    for (auto i = 0; i < rank; ++i) {
      ret.push_back(tensor_proto.dims(i));
    }
  }
  return ret;
}

const std::string& tensor_proto_get_name(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.name();
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i32(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT32);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int32_t));
  return tensor_proto;
}

// tensor proto for d3d12 (from cpu to gpu) - input tensor , add this before execution context
 ComPtr<ID3D12Resource> tensor_proto_new_d3d12_cpu_to_gpu(
    ID3D12Device* device,
    ComPtr<ID3D12Resource>& UploadBuffer,
    ID3D12GraphicsCommandList* cmdList,
    const void* initData,
    size_t byteSize
    ) {
    
    // copy surface to GPU buffer
    ComPtr<ID3D12Resource> GPUResource;
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);

    // Create the actual default buffer resource
    ORT_THROW_IF_FAILED(device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(GPUResource.GetAddressOf())));

    heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    // In order to copy CPU memory data into our default buffer, we need
    // to ORT_THROW_IF_FAILED an intermediate upload heap - it is GPU upload buffer 
    ORT_THROW_IF_FAILED(device->CreateCommittedResource(
         &heap,
         D3D12_HEAP_FLAG_NONE,
         &buffer,
         D3D12_RESOURCE_STATE_GENERIC_READ,
         nullptr,
         IID_PPV_ARGS(UploadBuffer.GetAddressOf())));

     // Describe the data we want to copy into the default buffer.
     
     // pData: A pointer to a system memory array which contains the data to initialize
     // the buffer with.If the buffer can store n vertices, then the system array must
     // contain at least n vertices so that the entire buffer can be initialized. 
     // RowPitch: For buffers, the size of the data we are copying in bytes
     // SlicePitch: For buffers, the size of the data we are copying in bytes.

     D3D12_SUBRESOURCE_DATA subResourceData = {};
     subResourceData.pData = initData;
     subResourceData.RowPitch = byteSize;
     subResourceData.SlicePitch = subResourceData.RowPitch;

     // Schedule to copy the data to the default buffer resource.
     // At a high level, the helper function UpdateSubresources
     // will copy the CPU memory into the intermediate upload heap.
     // Then, using ID3D12CommandList::CopySubresourceRegion,
     // the intermediate upload heap data will be copied to mBuffer.
     auto barrier1 = CD3DX12_RESOURCE_BARRIER::Transition(GPUResource.Get(),
                                                D3D12_RESOURCE_STATE_COMMON,
                                                D3D12_RESOURCE_STATE_COPY_DEST);
     cmdList->ResourceBarrier(1,&barrier1);
     UpdateSubresources<1>(cmdList,
                           GPUResource.Get(), UploadBuffer.Get(),
                           0, 0, 1, &subResourceData);

     auto barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(GPUResource.Get(),
                                                           D3D12_RESOURCE_STATE_COPY_DEST,
                                                           D3D12_RESOURCE_STATE_GENERIC_READ);
     cmdList->ResourceBarrier(1,&barrier2);
    
     return GPUResource;
}

void* tensor_proto_new_d3d12_gpu_to_cpu(
    const ComPtr<ID3D12Resource>& outputBuffer,
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    size_t tensorByteSize,
    ComPtr<ID3D12CommandQueue> cmdQueue) {

    // The readback buffer (created below) is on a readback heap, so that the CPU can access it.
    D3D12_HEAP_PROPERTIES readbackHeapProperties{CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)};
    D3D12_RESOURCE_DESC readbackBufferDesc{CD3DX12_RESOURCE_DESC::Buffer(tensorByteSize)};
    ComPtr<ID3D12Resource> readbackBuffer;
    ORT_THROW_IF_FAILED(device->CreateCommittedResource(
        &readbackHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &readbackBufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)
        ));

    // Schedule to copy the data to the default buffer to the readback
    // buffer.
    auto barrier1 = CD3DX12_RESOURCE_BARRIER::Transition(
        outputBuffer.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->ResourceBarrier(1, &barrier1);

    cmdList->CopyResource(readbackBuffer.Get(), outputBuffer.Get());

    auto barrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        outputBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON);
    cmdList->ResourceBarrier(1, &barrier2);

    // Wait for completion and map the result
    ORT_THROW_IF_FAILED(cmdList->Close());

    // Add the command list to the queue for execution.
    ID3D12CommandList* cmdsLists[] = {cmdList};
    cmdQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    FlushCommandQueue(cmdQueue, device);  

    // Map the readback heap and copy it into the destination
    D3D12_RANGE range{0, tensorByteSize};
    void* dst = malloc(tensorByteSize);
    void* bufferData = nullptr;
    
    ORT_THROW_IF_FAILED(readbackBuffer->Map(0, &range, reinterpret_cast<void**>(&bufferData)));    

    // copy the data into a system memory array for further processing on the CPU side
    memcpy(dst, bufferData, tensorByteSize);
   
    // unmap - deallocates cpu virtual address range
    D3D12_RANGE emptyRange{0, 0};
    readbackBuffer->Unmap(
        0,
        &emptyRange);

    return dst;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i64(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int64_t));
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i8(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int8_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT8);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int8_t));
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(float));
  return tensor_proto;
}

}  // namespace vaip
