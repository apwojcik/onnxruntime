/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    tanh.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in Eigen. Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"
#include "softmax.h"

//
// Bundles the floating point constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const struct {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
} MlasTanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

void
MLASCALL
MlasTanhKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N >= 4) {

        MLAS_FLOAT32X4 Value = MlasLoadFloat32x4(Input);

        Value = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasTanhConstants.LowerRange), Value);
        Value = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasTanhConstants.UpperRange), Value);

        MLAS_FLOAT32X4 ValueSquared = MlasMultiplyFloat32x4(Value, Value);

        MLAS_FLOAT32X4 p;
        p = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_13),
            MlasBroadcastFloat32x4(MlasTanhConstants.alpha_11));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_9));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_7));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_5));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_3));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_1));
        p = MlasMultiplyFloat32x4(p, Value);

        MLAS_FLOAT32X4 q;
        q = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_6),
            MlasBroadcastFloat32x4(MlasTanhConstants.beta_4));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_2));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_0));

        MlasStoreFloat32x4(Output, MlasDivideFloat32x4(p, q));

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {

        float Value = *Input++;

        // This odd two-step process exists to ensure an input value of NaN carries through
        // without modification because "std::min" and "std::max" return unreliable results
        // when NaNs are involved, and it's clear from the test's reference outputs that
        // they want a NaN on output whenever the input is a NaN.
        float v_tmp;
        v_tmp = (Value < MlasTanhConstants.LowerRange) ? MlasTanhConstants.LowerRange : Value;
        Value = (v_tmp > MlasTanhConstants.UpperRange) ? MlasTanhConstants.UpperRange : v_tmp;

        float ValueSquared = Value * Value;

        float p;
        p = ValueSquared * MlasTanhConstants.alpha_13 + MlasTanhConstants.alpha_11;
        p = p * ValueSquared + MlasTanhConstants.alpha_9;
        p = p * ValueSquared + MlasTanhConstants.alpha_7;
        p = p * ValueSquared + MlasTanhConstants.alpha_5;
        p = p * ValueSquared + MlasTanhConstants.alpha_3;
        p = p * ValueSquared + MlasTanhConstants.alpha_1;
        p = p * Value;

        float q;
        q = ValueSquared * MlasTanhConstants.beta_6 + MlasTanhConstants.beta_4;
        q = q * ValueSquared + MlasTanhConstants.beta_2;
        q = q * ValueSquared + MlasTanhConstants.beta_0;

        *Output++ = (p / q);

        N -= 1;
    }
}

template <>
void
MLASCALL
MlasComputeTanh<float>(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().TanhKernelRoutine(Input, Output, N);
#else
    MlasTanhKernel(Input, Output, N);
#endif
}

template <>
void
MLASCALL
MlasComputeTanh<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
) {
    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr || dispatch->Tanh_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Tanh_Fp16 is not supported.");
    }
    dispatch->Tanh_Fp16(Input, Output, N);
}

template <>
void
MLASCALL
MlasComputeSoftcap<float>(
    const float* Input,
    float* Output,
    size_t N,
    float cap
) {
    for (size_t i = 0; i < N; i++) {
        Output[i] = Input[i] / cap;
        Output[i] = std::tanh(Output[i]);
        Output[i] = Output[i] * cap;
    }
}

template <>
void
MLASCALL
MlasComputeSoftcap<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    MLAS_FP16 cap
) {
    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr || dispatch->Softcap_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Softcap_Fp16 is not supported.");
    }
    dispatch->Softcap_Fp16(Input, Output, N, cap);
}
