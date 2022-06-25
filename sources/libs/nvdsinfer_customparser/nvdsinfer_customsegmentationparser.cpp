/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include <cassert>
#include <cmath>


/* C-linkage to prevent name-mangling */

extern "C"
bool NvDsInferParseCustomDeplabv3plus (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferSegmentationOutput &output);

extern "C"
bool NvDsInferParseCustomDeplabv3plus (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferSegmentationOutput &output) {

    NvDsInferDimsCHW outputDimsCHW;
    getDimsHWCFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);

    output.width = outputDimsCHW.w;
    output.height = outputDimsCHW.h;
    output.classes = 1;

    output.class_map = new int [output.width * output.height];
    output.class_probability_map = (float*)outputLayersInfo[0].buffer;

    // printf("inferDims  numDims: %d, numElements: %d, d.[0]: %d, d.[1]: %d, d.[2]: %d , dataType %d \n", 
    //         outputLayersInfo[0].inferDims.numDims, outputLayersInfo[0].inferDims.numElements, outputLayersInfo[0].inferDims.d[0], outputLayersInfo[0].inferDims.d[1], outputLayersInfo[0].inferDims.d[2], outputLayersInfo[0].dataType);
    // printf("output.class_probability_map[1] %d [10] %d [10000] %d\n", output.class_probability_map[1], output.class_probability_map[262143], output.class_probability_map[262143]);
    int sky_count = 0;
    for (unsigned int y = 0; y < output.height; y++)
    {
        for (unsigned int x = 0; x < output.width; x++)
        {
            int &cls = output.class_map[y * output.width + x] = -1;
            cls = output.class_probability_map[0*output.width * output.height + y * output.width + x];
            // printf("%d ", cls);
            if (cls == 1) sky_count++;
        }
    }
    // printf("%d\n", sky_count);
    printf("sky rate is %d/%d=%f\n", sky_count, output.width * output.height, sky_count * 1.0/(output.width * output.height));

    return true;

}
/* Check that the custom function has been defined correctly */

CHECK_CUSTOM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDeplabv3plus);
