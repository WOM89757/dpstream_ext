/*
* Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __DEEPSTREAM_VC_APP_H_
#define __DEEPSTREAM_VC_APP_H_

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include "deepstream_asr_config_file_parser.h"

#define CHECK_PTR(ptr) \
  if(ptr ==  NULL) \
  { \
    return -1; \
  }

typedef struct __StreamCtx
{
  gchar *uri;
  guint stream_id;
  guint has_audio;
  guint bus_id;
  GstElement *asr_pipeline;
  int eos_received;
  NvDsAudioConfig audio_config;
  FILE *FP_asr;
} StreamCtx;

typedef struct __AppCtx
{
  guint num_sources;
  StreamCtx *sctx;
  NvDsAppConfig app_config;
} AppCtx;


int create_pipeline(AppCtx *appctx, int stream_num, StreamCtx *sctx);
int start_pipeline(int stream_num, StreamCtx *sctx);
int destroy_pipeline(StreamCtx *sctx);

guint get_num_sources (gchar *cfg_file_path);
gboolean parse_config_file (AppCtx *appctx, gchar *config_file);

#endif
