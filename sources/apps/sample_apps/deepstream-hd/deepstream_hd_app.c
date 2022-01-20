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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"


#define MAX_DISPLAY_LEN 64
#define MAX_TIME_STAMP_LEN 32


#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};



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

  dstMeta = g_memdup (srcMeta, sizeof(NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = g_memdup (srcMeta->objSignature.signature,
                                                srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if(srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj = (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
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
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

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
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
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
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
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
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

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

static void
generate_vehicle_meta (gpointer data)
{
  NvDsVehicleObject *obj = (NvDsVehicleObject *) data;

  obj->type = g_strdup ("sedan");
  obj->color = g_strdup ("blue");
  obj->make = g_strdup ("Bugatti");
  obj->model = g_strdup ("M");
  obj->license = g_strdup ("XX1234");
  obj->region = g_strdup ("CA");
}

static void
generate_person_meta (gpointer data)
{
  NvDsPersonObject *obj = (NvDsPersonObject *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel= g_strdup ("formal");
}

static void
generate_event_msg_meta (gpointer data, gint class_id, NvDsObjectMeta * obj_params)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  meta->sensorId = 0;
  meta->placeId = 0;
  meta->moduleId = 0;
  meta->sensorStr = g_strdup ("sensor-0");

  // if(srcMeta->objectId) {
  if(0) {
    meta->tagsStr = g_strdup ("{\"streamid\": \"12a\"}");
    // meta->tagsStr = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

    // strncpy(meta->tagsStr, "{\"streamid\": \"12a\"}", MAX_LABEL_SIZE);
    // strncpy(meta->tagsStr, obj_params->obj_label, MAX_LABEL_SIZE);
  }

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
    meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
    meta->objClassId = PGIE_CLASS_ID_VEHICLE;

    NvDsVehicleObject *obj = (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
    generate_vehicle_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsVehicleObject);
  } else if (class_id == PGIE_CLASS_ID_PERSON) {
    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON;
    meta->objClassId = PGIE_CLASS_ID_PERSON;

    NvDsPersonObject *obj = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
    generate_person_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsPersonObject);
  }
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
    gboolean is_first_object = TRUE;


    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        is_first_object = TRUE;
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);


        /*
        * Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
        * component implementing detection / recognition logic.
        * Here it demonstrates how to use / attach that meta data.
        */
        if (is_first_object && !(frame_number % 30)) {
          /* Frequency of messages to be send will be based on use case.
          * Here message is being sent for first object every frame_interval(default=30).
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

    // g_print ("Frame Number = %d Number of objects = %d "
    //         "Vehicle Count = %d Person Count = %d\n",
    //         frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

/* pgie_queue_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
typedef struct
{
  float left;
  float top;
  float width;
  float height;
  char label[10];
} DsExampleObject;
static GstPadProbeReturn
pgie_queue_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsObjectMeta *object_meta = NULL;


    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    NvDsFrameMeta *frame_meta = NULL;
    DsExampleObject obj = {5, 10, 800, 1300, 'r'};
    gdouble scale_ratio = 1.0;
    guint batch_id = 0;
    static gchar font_name[] = "Serif";

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) (l_frame->data);
        object_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        NvOSD_RectParams *rect_params = &object_meta->rect_params;
        NvOSD_TextParams *text_params = &object_meta->text_params;

        
        // NvOSD_RectParams & rect_params = object_meta->rect_params;
        // NvOSD_TextParams & text_params = object_meta->text_params;

        /* Assign bounding box coordinates */
        rect_params->left = obj.left;
        rect_params->top = obj.top;
        rect_params->width = obj.width;
        rect_params->height = obj.height;

        /* Semi-transparent yellow background */
        rect_params->has_bg_color = 0;
        rect_params->bg_color = (NvOSD_ColorParams) {
        1, 1, 0, 0.4};
        /* Red border of width 6 */
        rect_params->border_width = 3;
        rect_params->border_color = (NvOSD_ColorParams) {
        1, 0, 0, 1};

        /* Scale the bounding boxes proportionally based on how the object/frame was
        * scaled during input */
        rect_params->left /= scale_ratio;
        rect_params->top /= scale_ratio;
        rect_params->width /= scale_ratio;
        rect_params->height /= scale_ratio;
        // g_print ("Add region rect batch%u"
        //     "  left->%f top->%f width->%f"
        //     " height->%f label->%s\n", batch_id, rect_params->left,
        //     rect_params->top, rect_params->width, rect_params->height, obj.label);

        object_meta->object_id = UNTRACKED_OBJECT_ID;
        g_strlcpy (object_meta->obj_label, obj.label, MAX_LABEL_SIZE);
        /* display_text required heap allocated memory */
        text_params->display_text = g_strdup (obj.label);
        /* Display text above the left top corner of the object */
        text_params->x_offset = rect_params->left;
        text_params->y_offset = rect_params->top - 10;
        /* Set black background for the text */
        text_params->set_bg_clr = 1;
        text_params->text_bg_clr = (NvOSD_ColorParams) {
        0, 0, 0, 1};
        /* Font face, size and color */
        text_params->font_params.font_name = font_name;
        text_params->font_params.font_size = 11;
        text_params->font_params.font_color = (NvOSD_ColorParams) {
        1, 1, 1, 1};

        nvds_add_obj_meta_to_frame(frame_meta, object_meta, NULL);
        batch_id++;
    }
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
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL, *nvvidconv1 = NULL, *pgine_queue = NULL,
      *nvosd = NULL, *msgconv = NULL, *msgbroker, *msg_queue = NULL, *nvosd_queue = NULL, *tee0 = NULL;

  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL, *pgie_queue_sink_pad = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <H264 filename>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dshd-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-conv");
  msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");
  msg_queue = gst_element_factory_make ("queue", "msg_queue");
  nvosd_queue = gst_element_factory_make ("queue", "nvosd_queue");
  tee0 = gst_element_factory_make ("tee", "tee0");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }


  pgine_queue = gst_element_factory_make ("queue", "queue-primary-nvinference-engine");
  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
    g_printerr ("prop.integrated.\n");
  }else {
    g_printerr ("prop.integrated is 0.\n");
  }

  nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  sink = gst_element_factory_make ("fpsdisplaysink", "nvvideo-renderer");
  // sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!source || !h264parser || !decoder || !pgine_queue || !pgie
      || !nvvidconv || !nvvidconv1 || !nvosd || !msgconv || !msgbroker || !msg_queue || !nvosd_queue || !tee0 || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if(!transform && prop.integrated) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  g_object_set (G_OBJECT (source), "location", argv[1], NULL);

  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dshd_pgie_config.txt", 
      "infer-on-class-ids", "0",
      "process-mode", 2,
      NULL);
  g_object_set (G_OBJECT (msgconv),
      "config", "dshd_msgconv_config.txt", 
      "payload-type", 0,
      "msg2p-newapi", 0,
      "frame-interval", 30,
      NULL);

  g_object_set (G_OBJECT (msgbroker),
      "proto-lib", "/opt/nvidia/deepstream/deepstream/lib/libnvds_amqp_proto.so", 
      "conn-str", "192.168.56.103;5672;admin;admin",
      "sync", FALSE,
      "topic", "dstest",
      NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if(prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline),
        source, h264parser, decoder, streammux, pgine_queue, pgie,
        nvvidconv, nvosd, transform, sink, NULL);
  }
  else {
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgine_queue, pgie,
      nvvidconv, tee0, nvosd, msgconv, msgbroker, nvvidconv1, sink, msg_queue, nvosd_queue, NULL);
  }

  GstPad *sinkpad, *srcpad;
  GstPad *tee_msg_pad = NULL;
  GstPad *sink_pad = NULL;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if(prop.integrated) {
    if (!gst_element_link_many (streammux, pgine_queue, pgie,
        nvvidconv, nvosd, transform, sink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  }
  else {
    if (!gst_element_link_many (streammux, pgine_queue, pgie,
        nvvidconv, nvosd, tee0, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
    if (!gst_element_link_many (msg_queue, msgconv, msgbroker, NULL)) {
      g_printerr ("Elements could not be linked: 3. Exiting.\n");
      return -1;
    }
    if (!gst_element_link_many (nvosd_queue, nvvidconv1, sink, NULL)) {
      g_printerr ("Elements could not be linked: 4. Exiting.\n");
      return -1;
    }

    sink_pad = gst_element_get_static_pad (msg_queue, "sink");
    tee_msg_pad = gst_element_get_request_pad (tee0, "src_%u");
    if (!tee_msg_pad) {
      g_printerr ("Unable to get request pads\n");
      return -1;
    }

    if (gst_pad_link (tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Unable to link tee and message converter\n");
      gst_object_unref (sink_pad);
      return -1;
    }
    gst_object_unref (sink_pad);
    gst_object_unref (tee_msg_pad);

    sink_pad = gst_element_get_static_pad (nvosd_queue, "sink");
    tee_msg_pad = gst_element_get_request_pad (tee0, "src_%u");
    if (!tee_msg_pad) {
      g_printerr ("Unable to get request pads\n");
      return -1;
    }

    if (gst_pad_link (tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Unable to link tee and message converter\n");
      gst_object_unref (sink_pad);
      return -1;
    }
    gst_object_unref (sink_pad);
    gst_object_unref (tee_msg_pad);
  }

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

  pgie_queue_sink_pad = gst_element_get_static_pad (pgine_queue, "sink");
  if (!pgie_queue_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (pgie_queue_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_queue_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (pgie_queue_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
