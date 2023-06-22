// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./tensor_proto.h"
#include "./vai_assert.h"

#include <cstdint>
#include <limits>

namespace vaip {

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

// steps to do :
//  1. copy surface to linear accessible cpu buffer
//  2. Map it from cpu to gpu (copying texture from cpu to gpu)
//  3. copy data to tensor proto (copy input tensor)
//  4. unmap
// 
// tensor proto for d3d11 (from cpu to gpu) - input tensor , add this before execution context
ONNX_NAMESPACE::TensorProto tensor_proto_new_d3d12_cpu_to_gpu(
    const Microsoft::WRL::ComPtr<ID3D12Resource> InputBuffer,
    const ExecutionProviderImpl* provider,
    size_t byteSize
    ) {
    
    // according to DirectML example

    // 1. copy surface to GPU buffer

    // TODO:
    Microsoft::WRL::ComPtr<ID3D12Resource> GPUResource;

    // Create the actual default buffer resource.
    ORT_THROW_IF_FAILED(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(GPUResource.GetAddressOf())));

    // In order to copy CPU memory data into our default buffer, we need
    // to ORT_THROW_IF_FAILED an intermediate upload heap - it is GPu upload buffer
     ThrowIfFailed(device->CreateCommittedResource(
         &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
         D3D12_HEAP_FLAG_NONE,
         &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
         D3D12_RESOURCE_STATE_GENERIC_READ,
         nullptr,
         IID_PPV_ARGS(InputBuffer.GetAddressOf())));

     // Schedule to copy the data to the default buffer resource.
     // At a high level, the helper function UpdateSubresources
     // will copy the CPU memory into the intermediate upload heap.
     // Then, using ID3D12CommandList::CopySubresourceRegion,
     // the intermediate upload heap data will be copied to mBuffer.
     ID3D12GraphicsCommandList::ResourceBarrier(1,
                                                &CD3DX12_RESOURCE_BARRIER::Transition(GPUResource.Get(),
                                                                    D3D12_RESOURCE_STATE_COMMON,
                                                                    D3D12_RESOURCE_STATE_COPY_DEST));
     UpdateSubresources<1>(cmdList,
                           GPUResource.Get(), InputBuffer.Get(),
                           0, 0, 1, &subResourceData);
    ID3D12GraphicsCommandList::ResourceBarrier(1,
                                                &CD3DX12_RESOURCE_BARRIER::Transition(GPUResource.Get(),
                                                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                                                    D3D12_RESOURCE_STATE_GENERIC_READ));



    // add part CPU->GPU and what tyo return   
    
    return GPUResource;
}


// same thing for output tensor (from gpu to cpu) - this will be used after execution context
// steps:
// 1. copy surface to gpu buffer
// 2. map it from gpu to cpu
// 3. copy data to tensor proto
// 4. unmap

// surface is get from Execution Provider output

ONNX_NAMESPACE::TensorProto tensor_proto_new_d3d12_gpu_to_cpu(
    const Microsoft::WRL::ComPtr<ID3D12Resource> InputBuffer,
    const ExecutionProviderImpl* provider,
    size_t tensorByteSize) {
    // according to DirectML example and link from the direct3d 12 book there are following steps:

    //  1. create system memory buffer with heap properties 

    Microsoft::WRL::ComPtr<ID3D12Resource> OutputBuffer;

    D3D12_HEAP_PROPERTIES heapProperties = {
        D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0};

    D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                        0,
                                        static_cast<uint64_t>((tensorByteSize + 3) & ~3),
                                        1,
                                        1,
                                        1,
                                        DXGI_FORMAT_UNKNOWN,
                                        {1, 0},
                                        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

    Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
    ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

    ORT_THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(OutputBuffer.GetAddressOf())));


    //  2. use the ID3D12GraphicsCommandList::CopyResource method to copy the GPU resource to the system memory
    //    resource. system memory resource is the same type and size as resource from gpu

    // Schedule to copy the data to the default buffer to the readback
    ID3D12GraphicsCommandList::ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
                                                InputBuffer.Get(),
                                                D3D12_RESOURCE_STATE_COMMON,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE));
    ID3D12GraphicsCommandList::CopyResource(OutputBuffer.Get(), InputBuffer.Get());

    ID3D12GraphicsCommandList::ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
                                         InputBuffer.Get(),
                                         D3D12_RESOURCE_STATE_COPY_SOURCE,
                                         D3D12_RESOURCE_STATE_COMMON))
    //  3.  map the system memory buffer with the mapping API to read it on the CPU -  allocates cpu virtual range for the resource

    void* bufferData = nullptr;
    D3D12_RANGE range = {0, tensorByteSize};
    ORT_THROW_IF_FAILED(OutputBuffer->Map(0, &range, &bufferData));

    // 6. added to write into a file output

    std::ofstream fout("C:\\Users\\tvukovic\\dev.txt");
    for (int i = 0; i < NumDataElements; ++i) {
    fout << "(" << mappedData[i].v1.x << ", " << mappedData[i].v1.y << ", " << mappedData[i].v1.z << ", " << mappedData[i].v2.x << ", " << mappedData[i].v2.y << ")" << std::endl;
    }

    //or write to EP
    ORT_THROW_IF_FAILED(provider->UploadToResource(buffer.Get(), tensorPtr, tensorByteSize));

    //  4. copy the data into a system memory array for further processing on the CPU side
    
    memcpy(bufferData, tensorPtr, tensorByteSize);

    //  5. unmap - deallocates cpu virrtual address range
   
    buffer->Unmap(0, &range);

    return buffer;
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
