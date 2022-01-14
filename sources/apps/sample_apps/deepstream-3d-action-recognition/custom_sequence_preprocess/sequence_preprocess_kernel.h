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

#ifndef __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__
#define __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define VEC4_SIZE 4
// float vector structure for multiple channels
typedef struct {
    float d[VEC4_SIZE];
} Float4Vec;

/**
 * NCDHW preprocess per ROI image
 *
 * @param outPtr output data pointer offset to current image position
 * @param inPtr input data pointer
 */
cudaError_t preprocessNCDHW(
    void* outPtr, unsigned int outC, unsigned int H, unsigned int W, unsigned int S,
    const void* inPtr, unsigned int inC, unsigned int inRowPitch, Float4Vec scales, Float4Vec means,
    bool swapRB, cudaStream_t stream);

#endif  // __NVDS_SEQUENCE_PREPROCESS_KERNEL_H__