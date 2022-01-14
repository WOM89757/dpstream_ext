/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <iostream>

#include <math.h>
#include <gst/gst.h>
#include <glib.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/timeb.h>
#include <cuda_runtime_api.h>

/* Open CV headers */
#include "opencv2/imgproc/imgproc.hpp"

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

using namespace cv;
using namespace std;

#define MAX_DISPLAY_LEN 64
#define MAX_TIME_STAMP_LEN 32

#define PGIE_CLASS_ID_BG      0
#define PGIE_CLASS_ID_VEHICLE 1
#define PGIE_CLASS_ID_PERSON  2

#define PGIE_CONFIG_FILE  "dsmrcnn_pgie_config.txt"
#define MSCONV_CONFIG_FILE "dsmrcnn_msgconv_config.txt"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

static gchar *cfg_file = NULL;
static gchar *input_file = NULL;
static gchar *topic = NULL;
static gchar *conn_str = NULL;
static gchar *proto_lib = NULL;
static gint schema_type = 0;
static gboolean display_off = FALSE;

gint frame_number = 0;
gchar pgie_classes_str[3][32] = { "bg", "Vehicle", "Person" };

GOptionEntry entries[] = {
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfg_file,
   "Set the adaptor config file. Optional if connection string has relevant  details.", NULL},
  {"input-file", 'i', 0, G_OPTION_ARG_FILENAME, &input_file,
   "Set the input H264 file", NULL},
  {"topic", 't', 0, G_OPTION_ARG_STRING, &topic,
   "Name of message topic. Optional if it is part of connection string or config file.", NULL},
  {"conn-str", 0, 0, G_OPTION_ARG_STRING, &conn_str,
   "Connection string of backend server. Optional if it is part of config file.", NULL},
  {"proto-lib", 'p', 0, G_OPTION_ARG_STRING, &proto_lib,
   "Absolute path of adaptor library", NULL},
  {"schema", 's', 0, G_OPTION_ARG_INT, &schema_type,
   "Type of message schema (0=Full, 1=minimal), default=0", NULL},
  {"no-display", 0, 0, G_OPTION_ARG_NONE, &display_off, "Disable display", NULL},
  {NULL}
};

static void resizeMask(float *src, int original_width, int original_height, cv::Mat &dst, float threshold)
{
    auto clip = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
    auto target_height = dst.rows;
    auto target_width = dst.cols;
    float ratio_h = static_cast<float>(original_height) / static_cast<float>(target_height);
    float ratio_w = static_cast<float>(original_width) / static_cast<float>(target_width);
    int channel = 1;

    for (int y = 0; y < target_height; ++y)
    {
        for (int x = 0; x < target_width; ++x)
        {
            float x0 = static_cast<float>(x) * ratio_w;
            float y0 = static_cast<float>(y) * ratio_h;
            int left = static_cast<int>(clip(std::floor(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int top = static_cast<int>(clip(std::floor(y0), 0.0f, static_cast<float>(original_height - 1.0f)));
            int right = static_cast<int>(clip(std::ceil(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
            int bottom = static_cast<int>(clip(std::ceil(y0), 0.0f, static_cast<float>(original_height - 1.0f)));

            for (int c = 0; c < channel; ++c)
            {
                // H, W, C ordering
                float left_top_val = (float)src[top * (original_width * channel) + left * (channel) + c];
                float right_top_val = (float)src[top * (original_width * channel) + right * (channel) + c];
                float left_bottom_val = (float)src[bottom * (original_width * channel) + left * (channel) + c];
                float right_bottom_val = (float)src[bottom * (original_width * channel) + right * (channel) + c];
                float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
                float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
                float lerp = top_lerp + (bottom_lerp - top_lerp) * (y0 - top);
                if (lerp < threshold) {
                    dst.at<unsigned char>(y,x) = 0;
                } else {
                    dst.at<unsigned char>(y,x) = 255;
                }
            }
        }
    }
}

static void generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME,  &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec/1000000;
  g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static gpointer meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *)g_memdup (srcMeta, sizeof(NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *) g_memdup (srcMeta->objSignature.signature,
                                                srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if(srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE_EXT) {
      NvDsVehicleObjectExt *srcObj = (NvDsVehicleObjectExt *) srcMeta->extMsg;
      NvDsVehicleObjectExt *obj = (NvDsVehicleObjectExt *) g_malloc0 (sizeof (NvDsVehicleObjectExt));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObjectExt);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON_EXT) {
      NvDsPersonObjectExt *srcObj = (NvDsPersonObjectExt *) srcMeta->extMsg;
      NvDsPersonObjectExt *obj = (NvDsPersonObjectExt *) g_malloc0 (sizeof (NvDsPersonObjectExt));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);
      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObjectExt);
    }
  }

  return dstMeta;
}

static void meta_free_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  g_free (srcMeta->ts);
  g_free (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if(srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE_EXT) {
      NvDsVehicleObjectExt *obj = (NvDsVehicleObjectExt *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON_EXT) {
      NvDsPersonObjectExt *obj = (NvDsPersonObjectExt *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsgSize = 0;
  }
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

static GList *
generate_mask_polygon (float *mask, int mask_width, int mask_height,
                                    int rect_width, int rect_height,
                                    float threshold)
{
  GList *mask_list = NULL;
  vector<vector<cv::Point> > contours;
  cv::Mat dst = cv::Mat(rect_height, rect_width, CV_8UC1);

  resizeMask(mask, mask_width, mask_height, dst, threshold);

  findContours (dst, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  for( size_t i = 0; i< contours.size(); i++ ) {
    GArray *polygon;
    vector<cv::Point> temp = contours[i];
    polygon = g_array_new (FALSE, FALSE, sizeof(gdouble));

    for (size_t j = 0; j < temp.size(); j++) {
      gdouble xx = (gdouble)temp[j].x;
      gdouble yy = (gdouble)temp[j].y;
      g_array_append_val (polygon, xx);
      g_array_append_val (polygon, yy);
    }
    mask_list = g_list_append (mask_list, polygon);
  }
  return mask_list;
}

static void
generate_vehicle_meta (gpointer data, NvDsObjectMeta * obj_params)
{
  NvDsVehicleObjectExt *obj = (NvDsVehicleObjectExt *) data;

  obj->type = g_strdup ("sedan");
  obj->color = g_strdup ("blue");
  obj->make = g_strdup ("Bugatti");
  obj->model = g_strdup ("M");
  obj->license = g_strdup ("XX1234");
  obj->region = g_strdup ("CA");

  obj->mask = generate_mask_polygon (obj_params->mask_params.data,
                                     obj_params->mask_params.width,
                                     obj_params->mask_params.height,
                                     obj_params->rect_params.width,
                                     obj_params->rect_params.height,
                                     obj_params->mask_params.threshold);
}

static void
generate_person_meta (gpointer data, NvDsObjectMeta * obj_params)
{
  NvDsPersonObjectExt *obj = (NvDsPersonObjectExt *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel= g_strdup ("formal");

  obj->mask = generate_mask_polygon (obj_params->mask_params.data,
                                     obj_params->mask_params.width,
                                     obj_params->mask_params.height,
                                     obj_params->rect_params.width,
                                     obj_params->rect_params.height,
                                     obj_params->mask_params.threshold);
}

static void
generate_event_msg_meta (gpointer data, gint class_id, NvDsObjectMeta * obj_params)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  meta->sensorId = 0;
  meta->placeId = 0;
  meta->moduleId = 0;
  meta->sensorStr = g_strdup ("sensor-0");

  meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

  strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

  generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);

  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject. Then that object should
   * be handled in payload generator library (nvmsgconv.cpp) accordingly.
   */
  if (class_id == PGIE_CLASS_ID_VEHICLE) {
    meta->type = NVDS_EVENT_MOVING;
    meta->objType = NVDS_OBJECT_TYPE_VEHICLE_EXT;
    meta->objClassId = PGIE_CLASS_ID_VEHICLE;

    NvDsVehicleObjectExt *obj = (NvDsVehicleObjectExt *) g_malloc0 (sizeof (NvDsVehicleObjectExt));
    generate_vehicle_meta (obj, obj_params);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsVehicleObjectExt);
  } else if (class_id == PGIE_CLASS_ID_PERSON) {
    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON_EXT;
    meta->objClassId = PGIE_CLASS_ID_PERSON;

    NvDsPersonObjectExt *obj = (NvDsPersonObjectExt *) g_malloc0 (sizeof (NvDsPersonObjectExt));
    generate_person_meta (obj, obj_params);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsPersonObjectExt);
  }
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsFrameMeta *frame_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  gboolean is_first_object = TRUE;
  NvDsMetaList *l_frame, *l_obj;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (!batch_meta) {
    // No batch meta attached.
    return GST_PAD_PROBE_OK;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) l_frame->data;

    if (frame_meta == NULL) {
      // Ignore Null frame meta.
      continue;
    }

    is_first_object = TRUE;

    for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      if (obj_meta == NULL) {
        // Ignore Null object.
        continue;
      }

      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
        vehicle_count++;
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
        person_count++;

      /*
       * Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
       * component implementing detection / recognition logic.
       * Here it demonstrates how to use / attach that meta data.
       */
      if (is_first_object && !(frame_number % 30)) {
        /* Frequency of messages to be send will be based on use case.
         * Here message is being sent for first object every 30 frames.
         */

        NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
        msg_meta->bbox.top = obj_meta->rect_params.top;
        msg_meta->bbox.left = obj_meta->rect_params.left;
        msg_meta->bbox.width = obj_meta->rect_params.width;
        msg_meta->bbox.height = obj_meta->rect_params.height;
        msg_meta->frameId = frame_number;
        msg_meta->trackingId = obj_meta->object_id;
        msg_meta->confidence = obj_meta->confidence;
        generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta);

        NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta);
        if (user_event_meta) {
          user_event_meta->user_meta_data = (void *) msg_meta;
          user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
          user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) meta_copy_func;
          user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc) meta_free_func;
          nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
        } else {
          g_print ("Error in attaching event meta to buffer\n");
        }
        is_first_object = FALSE;
      }
    }
  }
  g_print ("Frame Number = %d "
      "Vehicle Count = %d Person Count = %d\n",
      frame_number, vehicle_count, person_count);
  frame_number++;

  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *nvstreammux;
  GstElement *msgconv = NULL, *msgbroker = NULL, *tee = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL;
  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstPad *tee_render_pad = NULL;
  GstPad *tee_msg_pad = NULL;
  GstPad *sink_pad = NULL;
  GstPad *src_pad = NULL;
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  ctx = g_option_context_new ("Nvidia DeepStream MaskRCNN");
  group = g_option_group_new ("MaskRCNN", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());

  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    g_option_context_free (ctx);
    g_printerr ("%s", error->message);
    return -1;
  }
  g_option_context_free (ctx);

  if (!proto_lib || !input_file) {
    g_printerr("missing arguments\n");
    g_printerr ("Usage: %s -i <H264 filename> -p <Proto adaptor library> --conn-str=<Connection string>\n", argv[0]);
    return -1;
  }

  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dsmrcnn-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  nvstreammux = gst_element_factory_make ("nvstreammux", "nvstreammux");

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Create msg converter to generate payload from buffer metadata */
  msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-converter");

  /* Create msg broker to send payload to server */
  msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");

  /* Create tee to render buffer and send message simultaneously*/
  tee = gst_element_factory_make ("tee", "nvsink-tee");

  /* Create queues */
  queue1 = gst_element_factory_make ("queue", "nvtee-que1");
  queue2 = gst_element_factory_make ("queue", "nvtee-que2");

  /* Finally render the osd output */
  if (display_off) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else {
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

    if(prop.integrated) {
        transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
        if (!transform) {
          g_printerr ("nvegltransform element could not be created. Exiting.\n");
          return -1;
        }
    }
  }

  if (!pipeline || !source || !h264parser || !decoder || !nvstreammux || !pgie
      || !nvvidconv || !nvosd || !msgconv || !msgbroker || !tee
      || !queue1 || !queue2 || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  g_object_set (G_OBJECT (source), "location", input_file, NULL);

  g_object_set (G_OBJECT (nvstreammux), "batch-size", 1, NULL);

  g_object_set (G_OBJECT (nvstreammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", PGIE_CONFIG_FILE, NULL);

  /** Enable display-mask;
   * Note: display-mask is supported only for process-mode=0 (CPU): */
  g_object_set (G_OBJECT(nvosd), "display-mask", TRUE, "process-mode", 0, NULL);

  g_object_set (G_OBJECT(msgconv), "config", MSCONV_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT(msgconv), "payload-type", schema_type, NULL);

  g_object_set (G_OBJECT(msgbroker), "proto-lib", proto_lib,
                "conn-str", conn_str, "sync", FALSE, NULL);

  if (topic) {
    g_object_set (G_OBJECT(msgbroker), "topic", topic, NULL);
  }

  if (cfg_file) {
    g_object_set (G_OBJECT(msgbroker), "config", cfg_file, NULL);
  }

  g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, nvstreammux, pgie,
      nvvidconv, nvosd, tee, queue1, queue2, msgconv,
      msgbroker, sink, NULL);

  if(prop.integrated) {
    if (!display_off)
      gst_bin_add (GST_BIN (pipeline), transform);
  }
  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder -> nvstreammux ->
   * nvinfer -> nvvidconv -> nvosd -> tee -> video-renderer
   *                                      |
   *                                      |-> msgconv -> msgbroker  */

  sink_pad = gst_element_get_request_pad (nvstreammux, "sink_0");
  if (!sink_pad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  src_pad = gst_element_get_static_pad (decoder, "src");
  if (!src_pad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (src_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sink_pad);
  gst_object_unref (src_pad);

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (nvstreammux, pgie, nvvidconv, nvosd, tee, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (queue1, msgconv, msgbroker, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if(prop.integrated) {
    if (!display_off) {
      if (!gst_element_link_many (queue2, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    } else {
      if (!gst_element_link (queue2, sink)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }
  }
  else {
    if (!gst_element_link (queue2, sink)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  sink_pad = gst_element_get_static_pad (queue1, "sink");
  tee_msg_pad = gst_element_get_request_pad (tee, "src_%u");
  tee_render_pad = gst_element_get_request_pad (tee, "src_%u");
  if (!tee_msg_pad || !tee_render_pad) {
    g_printerr ("Unable to get request pads\n");
    return -1;
  }

  if (gst_pad_link (tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and message converter\n");
    gst_object_unref (sink_pad);
    return -1;
  }

  gst_object_unref (sink_pad);

  sink_pad = gst_element_get_static_pad (queue2, "sink");
  if (gst_pad_link (tee_render_pad, sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and render\n");
    gst_object_unref (sink_pad);
    return -1;
  }

  gst_object_unref (sink_pad);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", input_file);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");

  g_free (cfg_file);
  g_free (input_file);
  g_free (topic);
  g_free (conn_str);
  g_free (proto_lib);

  /* Release the request pads from the tee, and unref them */
  gst_element_release_request_pad (tee, tee_msg_pad);
  gst_element_release_request_pad (tee, tee_render_pad);
  gst_object_unref (tee_msg_pad);
  gst_object_unref (tee_render_pad);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
