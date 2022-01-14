/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __DEEPSTREAM_ACTION_H__
#define __DEEPSTREAM_ACTION_H__

#include <cuda_runtime_api.h>

#include <fstream>
#include <functional>
#include <gst/gst.h>
#include <glib.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <queue>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "gstnvdsinfer.h"

#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

/* Print log message*/
#define LOG(out, level, fmt, ...)  \
    fprintf(out, "[%s: DS_3DAR] " fmt "\n", #level, ##__VA_ARGS__)

/* Print Debug message if ENABLE_DEBUG not zero */
#define LOG_DEBUG(fmt, ...)                     \
    if (gActionConfig.debug >= kDebugEnable) {  \
      LOG(stdout, DEBUG, fmt, ##__VA_ARGS__);   \
    }

/* Print Error message*/
#define LOG_ERROR(fmt, ...)  \
    LOG(stderr, ERROR, fmt, ##__VA_ARGS__)

template<typename T>
class SafePtr: public std::unique_ptr<T, std::function<void(T*)>> {
public:
  SafePtr(T* p, std::function<void(T*)> freefn):
    std::unique_ptr<T, std::function<void(T*)>>(p, freefn)
  {}
};

enum DebugLevel {
  kDebugDisable = 0,
  kDebugEnable,
  kDebugVerbose,
};

// FPS calculation for each source stream
class FpsCalculation {
public:
  FpsCalculation(uint32_t interval)
    : _max_frame_nums(interval)
  {}
  float updateFps(uint32_t source_id) {
    struct timeval time_now;
    gettimeofday(&time_now, nullptr);
    double now = (double)time_now.tv_sec + time_now.tv_usec / (double)1000000; // second
    float fps = -1.0f;
    auto iSrc = _timestamps.find(source_id);
    if (iSrc != _timestamps.end()) {
      auto & tms = iSrc->second;
      fps = tms.size() / (now - tms.front());
      while (tms.size() >= _max_frame_nums) {
        tms.pop();
      }
    } else {
      iSrc = _timestamps.emplace(source_id, std::queue<double>()).first;
    }
    iSrc->second.push(now);

    return fps;
  }
private:
  std::unordered_map<uint32_t, std::queue<double>> _timestamps;
  uint32_t _max_frame_nums = 50;
};

struct NvDsARConfig
{
  // stream source list
  std::vector<std::string> uri_list;

  // display sink settings
  gboolean display_sync = true;

  // nvdspreprocess plugin config file path
  std::string preprocess_config;
  // nvinfer plugin config file path
  std::string infer_config;

  // nvstreammux settings
  uint32_t muxer_height = 720;
  uint32_t muxer_width = 1280;
  // batched-push-timeout in usec, default value 40ms
  int32_t muxer_batch_timeout = 40000;

  // tiler settings
  uint32_t tiler_height = 720;
  uint32_t tiler_width = 1280;

  // debug level, disabled by default
  DebugLevel debug = kDebugDisable;

  // enable fps print on screen. enabled by default
  gboolean enableFps = TRUE;
};

// parse action recognition config into NvDsARConfig
bool parse_action_config(const char* action_config_path, NvDsARConfig& config);

#endif