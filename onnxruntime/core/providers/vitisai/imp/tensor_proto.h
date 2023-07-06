// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/gsl.h"
#include "onnx/onnx_pb.h"
#include "core/providers/dml/DmlExecutionProvider/src/External/D3DX12/d3dx12.h"
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"

#include <cstdint>
#include <limits>
#include <wrl.h>
#include <wil/Resource.h>
#include <wrl.h>
#include <d3d12.h>

#define FENCE_SIGNAL_VALUE 1


namespace vaip {
using Microsoft::WRL::ComPtr;

void FlushCommandQueue(ComPtr<ID3D12CommandQueue> cmdQueue);

gsl::span<const char> tensor_proto_as_raw(
    const ONNX_NAMESPACE::TensorProto& tensor);
size_t tensor_proto_raw_data_size(const ONNX_NAMESPACE::TensorProto& tensor);

std::vector<int64_t> tensor_proto_get_shape(
    const ONNX_NAMESPACE::TensorProto& tensor);
const std::string& tensor_proto_get_name(
    const ONNX_NAMESPACE::TensorProto& tensor);
ONNX_NAMESPACE::TensorProto tensor_proto_new_i8(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int8_t>& data);
ONNX_NAMESPACE::TensorProto tensor_proto_new_i32(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data);

ComPtr<ID3D12Resource> tensor_proto_new_d3d12_cpu_to_gpu(  //
    ID3D12Device* device,
    ComPtr<ID3D12Resource> UploadBuffer,
    ID3D12GraphicsCommandList* cmdList,
    const void* initData,
    size_t byteSize);

void* tensor_proto_new_d3d12_gpu_to_cpu(
    const ComPtr<ID3D12Resource> InputBuffer,
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    size_t tensorByteSize,
    ComPtr<ID3D12CommandQueue> cmdQueue);

ONNX_NAMESPACE::TensorProto tensor_proto_new_i64(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& data);

ONNX_NAMESPACE::TensorProto tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data);

}  // namespace vaip
