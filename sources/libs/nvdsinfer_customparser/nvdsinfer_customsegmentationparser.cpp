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
#include <map>
#include <sstream>


/* C-linkage to prevent name-mangling */

extern "C"
bool NvDsInferParseCustomDeeplabv3plus (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferTensorOrder &m_SegmentationOutputOrder,
                                   NvDsInferSegmentationOutput &output);

extern "C"
bool NvDsInferParseCustomDeeplabv3plus (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferTensorOrder &m_SegmentationOutputOrder,
                                   NvDsInferSegmentationOutput &output) {

    NvDsInferDimsCHW outputDimsCHW;
    if (m_SegmentationOutputOrder == NvDsInferTensorOrder_kNCHW) {
        getDimsCHWFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);
    }
    else if (m_SegmentationOutputOrder == NvDsInferTensorOrder_kNHWC) {
        getDimsHWCFromDims(outputDimsCHW, outputLayersInfo[0].inferDims);
    }

    output.width = outputDimsCHW.w;
    output.height = outputDimsCHW.h;
    output.classes = 1;

    // output.class_map = new int [output.width * output.height];
    // output.class_map = new int [output.width * output.height];
    // output.class_probability_map = new float [output.width * output.height];
    // output.class_map = (int*)outputLayersInfo[0].buffer;

    output.class_map = new int [output.width * output.height];
    int *result_m = (int*)outputLayersInfo[0].buffer;
    output.class_probability_map = (float*)outputLayersInfo[0].buffer;
    

    // printf("inferDims  numDims: %d, numElements: %d, d.[0]: %d, d.[1]: %d, d.[2]: %d , dataType %d \n", 
    //         outputLayersInfo[0].inferDims.numDims, outputLayersInfo[0].inferDims.numElements, outputLayersInfo[0].inferDims.d[0], outputLayersInfo[0].inferDims.d[1], outputLayersInfo[0].inferDims.d[2], outputLayersInfo[0].dataType);
    // printf("output.class_probability_map[1] %d [10] %d [10000] %d\n", output.class_probability_map[1], output.class_probability_map[262143], output.class_probability_map[262143]);

    int sky_count = 0, tree_count = 0, road_count = 0;
    std::map<int, int> result_map;

    for (unsigned int y = 0; y < output.height; y++)
    {
        for (unsigned int x = 0; x < output.width; x++)
        {
            int &cls = output.class_map[y * output.width + x];
            cls = result_m[y * output.width + x];
            // int &cls = output.class_map[y * output.width + x] = -1;
            // cls = output.class_probability_map[y * output.width + x];
            // printf("%d ", cls);
            // if (cls == 1) sky_count++;
            if (result_map.count(cls)) {
                result_map[cls]++;
            } else {
                result_map[cls] = 1;
            }
            if (cls == 2) sky_count++;
            if (cls == 4) tree_count++;
            if (cls == 6) road_count++;

        }
    }

    std::stringstream map_stream;
    for (auto iter = result_map.begin(); iter != result_map.end(); iter++) {
        map_stream << iter->first << ":" << iter->second << "  ";
    }

    // printf("%d\n", sky_count);
    printf("result map is :\t %s\n", map_stream.str().c_str());
    printf("sky rate is %d/%d=%f\n", sky_count, output.width * output.height, sky_count * 1.0/(output.width * output.height));
    printf("tree rate is %d/%d=%f\n", tree_count, output.width * output.height, tree_count * 1.0/(output.width * output.height));
    printf("road rate is %d/%d=%f\n", road_count, output.width * output.height, road_count * 1.0/(output.width * output.height));

    return true;

}
/* Check that the custom function has been defined correctly */

CHECK_CUSTOM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDeeplabv3plus);
