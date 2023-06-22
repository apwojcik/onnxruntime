// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
//
// #include "..\..\..\..\..\include\onnxruntime\core\common"
//#include "onnx/onnx_pb.h"

namespace vaip {

//gsl::span<const char> tensor_proto_as_raw(
//    const ONNX_NAMESPACE::TensorProto& tensor);
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


ONNX_NAMESPACE::TensorProto tensor_proto_new_d3d12_cpu_to_gpu(
    const Microsoft::WRL::ComPtr<ID3D12Resource> InputBuffer,
    const ExecutionProviderImpl* provider,
    size_t byteSize);

ONNX_NAMESPACE::TensorProto tensor_proto_new_d3d12_gpu_to_cpu(
    const Microsoft::WRL::ComPtr<ID3D12Resource> InputBuffer,
    const ExecutionProviderImpl* provider,
    size_t tensorByteSize);


ONNX_NAMESPACE::TensorProto tensor_proto_new_i64(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& data);

ONNX_NAMESPACE::TensorProto tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data);

}  // namespace vaip
