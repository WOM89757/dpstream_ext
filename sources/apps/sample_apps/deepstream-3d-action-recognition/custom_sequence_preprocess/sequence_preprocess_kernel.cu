/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>

#include "sequence_preprocess_kernel.h"

#define THREADS_PER_BLOCK_W 16
#define THREADS_PER_BLOCK_H 8

// device functions
__device__ __constant__ unsigned int kSwapChannels[4] = {2, 1, 0, 3};

template <bool swap>
__device__ __forceinline__ unsigned int inChannel(unsigned int i);

template <>
__device__ __forceinline__ unsigned int
inChannel<false>(unsigned int i)
{
    return i;
}

template <>
__device__ __forceinline__ unsigned int
inChannel<true>(unsigned int i)
{
    return kSwapChannels[i];
}

/**
 * convert HWC to NCDHW preprocess with normalization
 */
template <typename OutT, typename InT, bool swapRB>
__global__ void
ImageHWCToCSHW(
    OutT* out, unsigned int C, unsigned int H, unsigned int W, unsigned int SHW, const InT* in,
    unsigned int inC, unsigned int inRowPitch, Float4Vec mult, Float4Vec plus)
{
    unsigned int c = threadIdx.x;
    unsigned int w = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int h = blockIdx.y * blockDim.z + threadIdx.z;

    if (w >= W || h >= H) {
        return;
    }
    unsigned int inIdx = h * inRowPitch + w * inC + inChannel<swapRB>(c);
    float inData = 0.0f;
    if (c < inC) {
        inData = (float)in[inIdx];
    }
    unsigned int outIdx = c * SHW + h * W + w;
    float val = inData * mult.d[c] + plus.d[c];
    out[outIdx] = val;
}

/**
 * NCDHW preprocess host function
 */
cudaError_t
preprocessNCDHW(
    void* outPtr, unsigned int outC, unsigned int H, unsigned int W, unsigned int S,
    const void* inPtr, unsigned int inC, unsigned int inRowPitch, Float4Vec scales, Float4Vec means,
    bool swapRB, cudaStream_t stream)
{
    unsigned int HW = H * W;
    unsigned int SHW = S * HW;

    Float4Vec mult = scales;
    Float4Vec plus;
    for (int i = 0; i < VEC4_SIZE; ++i) {
        plus.d[i] = -scales.d[i] * means.d[i];
    }

    dim3 blocks(outC, THREADS_PER_BLOCK_W, THREADS_PER_BLOCK_H);  // (C, W, H)
    // grids (W_block, H_block, 1)
    dim3 grids(
        (W + THREADS_PER_BLOCK_W - 1) / THREADS_PER_BLOCK_W,
        (H + THREADS_PER_BLOCK_H - 1) / THREADS_PER_BLOCK_H, 1);

    if (swapRB) {
        ImageHWCToCSHW<float, unsigned char, true><<<grids, blocks, 0, stream>>>(
            (float*)outPtr, outC, H, W, SHW, (const unsigned char*)inPtr, inC, inRowPitch, mult,
            plus);
    } else {
        ImageHWCToCSHW<float, unsigned char, false><<<grids, blocks, 0, stream>>>(
            (float*)outPtr, outC, H, W, SHW, (const unsigned char*)inPtr, inC, inRowPitch, mult,
            plus);
    }
    return cudaGetLastError();
}