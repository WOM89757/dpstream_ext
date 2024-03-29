/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <gst/gstpipeline.h>

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#define PGIE_CONFIG_FILE  "perf_demo_pgie_config.txt"
#define SGIE1_CONFIG_FILE "perf_demo_sgie1_config.txt"
#define SGIE2_CONFIG_FILE "perf_demo_sgie2_config_yolov5_onnx.txt"

// #define SGIE1_CONFIG_FILE "perf_demo_sgie2_config.txt"
// #define SGIE1_CONFIG_FILE "perf_demo_sgie2_config_yolov5_onnx.txt"
// #define SGIE2_CONFIG_FILE "perf_demo_sgie1_config.txt"
// #define SGIE2_CONFIG_FILE "perf_demo_sgie2_config.txt"
#define SGIE3_CONFIG_FILE "perf_demo_sgie3_config.txt"
#define MSCONV_CONFIG_FILE "dstest_pipeline1_msgconv_config.txt"

#define MAX_TIME_STAMP_LEN 32

#define GPU_ID 0

#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0
#define SGIE0_CLASS_ID_HEAD 0
// #define SGIE1_CLASS_ID_UNSAFETYBELT 20
#define SGIE1_CLASS_ID_UNSAFETYBELT 12
#define MAX_DISPLAY_LEN 64

#define PGIE_DETECTOR_UID 1
#define SGIE1_CLASSIFIER_UID 2
#define SGIE2_DETECTOR_UID 3

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
// #define MUXER_OUTPUT_WIDTH 1280
// #define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

guint sgie1_unique_id = 2;
guint sgie2_unique_id = 3;
guint sgie3_unique_id = 4;

static gchar *cfg_file = NULL;
static gchar *input_file = NULL;
static gchar *topic = NULL;
static gchar conn_str[] = "192.168.3.18;5672;guest;guest";
static gchar proto_lib[] = "../../../../lib/libnvds_amqp_proto.so";
static gint schema_type = 0;
static gint msg2p_meta = 0;
static gint frame_interval = 30;

gint frame_number = 0;
// gchar pgie_classes_str[4][32] = { "person", "", "Person",
//   "Roadsign"
// };
gchar pgie_classes_str[8][32] = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck"
};

static GMainLoop* loop = NULL;
std::vector<std::string> file_list;

#if defined(ENABLE_PROFILING)
static gint frame_number = 0;
static struct timeval g_start;
static struct timeval g_end;
static float g_accumulated_time_macro = 0;

static void profile_start() {
    gettimeofday(&g_start, 0);
}

static void profile_end() {
    gettimeofday(&g_end, 0);
}

static void profile_result() {
    g_accumulated_time_macro += 1000000 * (g_end.tv_sec - g_start.tv_sec)
                                + g_end.tv_usec - g_start.tv_usec;
    // Be careful 1000000 * g_accumulated_time_macro may be overflow.
    float fps = (float)((frame_number - 100) / (float)(g_accumulated_time_macro / 1000000));
    std::cout << "The average frame rate is " << fps
              << ", frame num " << frame_number - 100
              << ", time accumulated " << g_accumulated_time_macro/1000000
              << std::endl;
}
#endif

static char *getOneFileName(DIR *pDir, int &isFile) {
    struct dirent *ent;

    while (1) {
        ent = readdir(pDir);
        if (ent == NULL) {
            isFile = 0;
            return NULL;
        } else {
            if(ent->d_type & DT_REG) {
                isFile = 1;
                return ent->d_name;
            } else if (strcmp(ent->d_name, ".") == 0 ||
                       strcmp(ent->d_name, "..") == 0) {
                continue;
            } else {
                isFile = 0;
                return ent->d_name;
            }
        }
    }
}

static void get_file_list(char* inputDir) {
    if (inputDir == NULL) return;

    char *fn = NULL;
    int isFile = 1;
    std::string fnStd;
    std::string dirNameStd(inputDir);
    std::string fullName;

    DIR *dir = opendir(inputDir);

    while (1) {
        fn = getOneFileName(dir, isFile);

        if (isFile) {
            fnStd = fn;
            fullName = dirNameStd + "/" + fnStd;
            file_list.push_back(fullName);
        } else {
            break;
        }
    }
}

static gboolean source_switch_thread(gpointer* data) {
    g_print("change ----------- source \n ");
    static guint stream_num = 1;
    const char* location = file_list[stream_num % file_list.size()].c_str();

    GstElement* pipeline = (GstElement*) data;
    GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "file-source");
    GstElement* h264parser = gst_bin_get_by_name(GST_BIN(pipeline), "decodebin-decoder");
    // GstElement* h264parser = gst_bin_get_by_name(GST_BIN(pipeline), "h264-parser");
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "nvvideo-renderer");
    gst_element_set_state(pipeline, GST_STATE_PAUSED);
    GstStateChangeReturn ret = GST_STATE_CHANGE_FAILURE;
    ret = gst_element_set_state(source, GST_STATE_NULL);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_print("Unable to set state change for source element \n");
        g_main_loop_quit(loop);
    }

    g_object_set(G_OBJECT(source), "location", location, NULL);
    gst_pad_activate_mode(gst_element_get_static_pad(h264parser, "sink"), GST_PAD_MODE_PUSH, TRUE);
    gst_element_sync_state_with_parent(h264parser);
    gst_element_sync_state_with_parent(source);
    gst_element_sync_state_with_parent(sink);

#if 0 // Change rows/colums dynamically here
    guint rows = 0;
    guint columns = 0;
    g_object_get(G_OBJECT(sink), "rows", &rows, NULL);
    g_object_get(G_OBJECT(sink), "columns", &columns, NULL);

    if (stream_num % (rows * columns) == 0) {
        g_object_set (G_OBJECT(sink), "rows", rows * 2, NULL);
        g_object_set (G_OBJECT(sink), "columns", columns * 2, NULL);
    }
#endif
    stream_num++;

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    return FALSE;
}

static GstPadProbeReturn eos_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    gboolean ret = TRUE;
    GstEvent *event = GST_EVENT (info->data);

    static guint64 prev_accumulated_base = 0;
    static guint64 accumulated_base = 0;

    if ((info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        GST_BUFFER_PTS(GST_BUFFER(info->data)) += prev_accumulated_base;
    }

    if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)) {
      g_print("--------------eos probe cb event  type %d \n", GST_EVENT_TYPE(event));
        if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
            g_print("--------------eos event\n");
            ret = gst_element_seek((GstElement*) u_data,
                                   1.0,
                                   GST_FORMAT_TIME,
                                   (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
                                   GST_SEEK_TYPE_SET,
                                   0,
                                   GST_SEEK_TYPE_NONE,
                                   GST_CLOCK_TIME_NONE);
            if (!ret) {
                g_print("###Error in seeking pipeline\n");
            }
            g_idle_add((GSourceFunc) source_switch_thread, u_data);
        }
    }

    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
        GstSegment *segment;

        gst_event_parse_segment (event, (const GstSegment **) &segment);
        segment->base = accumulated_base;
        prev_accumulated_base = accumulated_base;
        accumulated_base += segment->stop;
    }

    switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
    /* QOS events from downstream sink elements cause decoder to drop
     * frames after looping the file since the timestamps reset to 0.
     * We should drop the QOS events since we have custom logic for
     * looping individual sources. */
    case GST_EVENT_QOS:
    case GST_EVENT_SEGMENT:
        return GST_PAD_PROBE_DROP;
    default:
        break;
    }

    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    GMainLoop* loop = (GMainLoop*) data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar* debug;
        GError* error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        g_free(debug);
        g_printerr("Error: %s\n", error->message);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
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
    dstMeta->objSignature.signature = (gdouble *)g_memdup (srcMeta->objSignature.signature,
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
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_VIOLATION) {
      NvDsViolationObject *srcObj = (NvDsViolationObject *) srcMeta->extMsg;
      NvDsViolationObject *obj = (NvDsViolationObject *) g_malloc0(sizeof (NvDsViolationObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->imagePath)
        obj->imagePath = g_strdup (srcObj->imagePath);
      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsViolationObject);
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
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_VIOLATION) {
      NvDsViolationObject *obj = (NvDsViolationObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->license)
        g_free (obj->license);
      if (obj->imagePath)
        g_free (obj->imagePath);
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
generate_violation_meta (gpointer data)
{
  NvDsViolationObject *obj = (NvDsViolationObject *) data;

  obj->type = g_strdup ("unsaf");
  obj->license = g_strdup ("XX1234");
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
generate_event_msg_meta (gpointer data, gint class_id, NvDsObjectMeta * obj_params, const gchar * save_image_path)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  meta->sensorId = 0;
  meta->placeId = 1;
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
  if (class_id == SGIE1_CLASS_ID_UNSAFETYBELT) {
    meta->type = NVDS_EVENT_UNSAFETY_BELT;
    meta->objType = NVDS_OBJECT_TYPE_VIOLATION;
    meta->objClassId = PGIE_CLASS_ID_VEHICLE;

    NvDsViolationObject *obj = (NvDsViolationObject *) g_malloc0 (sizeof (NvDsViolationObject));
    obj->type = g_strdup ("unsafety-belt");
    obj->license = g_strdup ("");
    obj->imagePath = g_strdup(save_image_path);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsViolationObject);
  }
}


typedef struct
{
  float left;
  float top;
  float width;
  float height;
  char label[10];
} DsExampleObject;




static GstPadProbeReturn
osd_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer ctx)
{
    GstBuffer *buf = (GstBuffer *) info->data;

    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
      GST_ERROR ("input buffer mapinfo failed");
      // return GST_FLOW_ERROR;
      return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
    gst_buffer_unmap (buf, &inmap);


    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint safety_count = 0;
    guint num_rects = 0;
    guint lable_i = 0;
    gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj, *l_class, *l_label;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsClassifierMeta * class_meta = NULL;
    NvDsLabelInfo * label_info = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
        // No batch meta attached.
        return GST_PAD_PROBE_OK;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;

        // g_print("osd src frame meta num is %d\n", frame_meta->num_obj_meta);
        if (frame_meta == NULL) {
          // Ignore Null frame meta.
          continue;
        }
        // continue;
        is_first_object = TRUE;

        for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            if (obj_meta == NULL) {
              // Ignore Null object.
              continue;
            }
            // if (obj_meta->unique_component_id != SGIE2_DETECTOR_UID)
            //   continue;
            // g_print("%d\n ", (frame_number % frame_interval));
            if ((is_first_object && !(frame_number % 29))) {
              g_print("here %d\n", frame_number);

              const char *osd_string = "OSD_SRC";
              char fileNameStringFull[FILE_NAME_SIZE];
              const char *save_full_img_path = "/opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-perf-test-pipeline1/full_result_imgs_img/";
              snprintf (fileNameStringFull, FILE_NAME_SIZE, "%s%s_%d_%d_%d_%s_%dx%d.jpg",
                  save_full_img_path,
                  osd_string, frame_number, frame_meta->source_id, 1,
                  obj_meta->obj_label, ip_surf->surfaceList[frame_meta->batch_id].width, ip_surf->surfaceList[frame_meta->batch_id].height);

              g_print("---------------save image buffer jto file %s\n", fileNameStringFull);

              NvDsObjEncUsrArgs userData = { 0 };
              /* To be set by user */
              userData.saveImg = TRUE;
              userData.attachUsrMeta = FALSE;
              /* Set if Image scaling Required */
              userData.scaleImg = FALSE;
              userData.scaledWidth = 0;
              userData.scaledHeight = 0;
              /* Preset */
              userData.objNum = 1;
              /* Quality */
              userData.quality = 80;
              // userData.fileNameImg = &fileNameString;
              memcpy(userData.fileNameImg, fileNameStringFull, sizeof(fileNameStringFull));
              // g_print("file name is %s\n", userData.fileNameImg);

              // memcpy(userData.fileNameImg, fileNameStringFull, sizeof(fileNameStringFull));
              unsigned dummy_counter = 0;
              /// Creating a special object meta in order to save a full frame
              NvDsObjectMeta dummy_obj_meta;
              dummy_obj_meta.rect_params.width = ip_surf->surfaceList[frame_meta->batch_id].width;
              dummy_obj_meta.rect_params.height = ip_surf->surfaceList[frame_meta->batch_id].height;
              dummy_obj_meta.rect_params.top = 0;
              dummy_obj_meta.rect_params.left = 0;
              nvds_obj_enc_process((NvDsObjEncCtxHandle)ctx, &userData, ip_surf, &dummy_obj_meta, frame_meta);
              is_first_object = FALSE;
          }
        }
        frame_number++;
    }

    nvds_obj_enc_finish((NvDsObjEncCtxHandle)ctx);
    return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint safety_count = 0;
    guint num_rects = 0;
    guint lable_i = 0;
    gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj, *l_class, *l_label;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsClassifierMeta * class_meta = NULL;
    NvDsLabelInfo * label_info = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
        // No batch meta attached.
        return GST_PAD_PROBE_OK;
    }


    DsExampleObject example_obj = {5, 10, 800, 1300, 'r'};
    gdouble scale_ratio = 1.0;

    // g_print("batch frame meta num is %d\n", batch_meta->num_frames_in_batch);
    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;

        // g_print("osd frame meta num is %d\n", frame_meta->num_obj_meta);
        if (frame_meta == NULL) {
          // Ignore Null frame meta.
          continue;
        }
        continue;
        is_first_object = TRUE;

        for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            if (obj_meta == NULL) {
              // Ignore Null object.
              continue;
            }

            // if (obj_meta->class_id == 110) {
            //   g_print("osd 110 parnet is %p\n", obj_meta->parent);
            // }

            txt_params = &(obj_meta->text_params);
            // if (txt_params->display_text)
            //   g_free (txt_params->display_text);

            // txt_params->display_text = (char*)g_malloc0 (MAX_DISPLAY_LEN);

            // g_snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "%s ",
            //             pgie_classes_str[obj_meta->class_id]);

            // g_print("unique component id is %d\n", obj_meta->unique_component_id);
            // g_print("objct traker id %lu\n ", obj_meta->object_id);
            if (obj_meta->unique_component_id == PGIE_DETECTOR_UID) {
                if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
                    vehicle_count++;
                if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
                    person_count++;
            }
            if (obj_meta->unique_component_id == SGIE1_CLASSIFIER_UID) {
                    g_print("head\n");
                if (obj_meta->class_id == 0) {
                    if (obj_meta->parent) {
                        g_print ("Head found for parent object %p (type=%s)\n",
                          obj_meta->parent,  obj_meta->parent->obj_label);
                    }

                }
            }

            if (obj_meta->unique_component_id == SGIE2_DETECTOR_UID) {
              // g_print("obj parent %p  current %p left %f \n", obj_meta->parent, obj_meta, obj_meta->detector_bbox_info.org_bbox_coords.left);
              // g_print("");
              // g_print("safety detect result is %s(%d)\n", obj_meta->obj_label, obj_meta->class_id);
              if (obj_meta->class_id == 12) {
                    safety_count++;
                    num_rects++;
                    // g_print("safety detect result is %s(%d) conf %f\n", obj_meta->obj_label, obj_meta->class_id, obj_meta->confidence);
                    // g_print(" is %d\n ", NVDS_EVENT_CUSTOM);
                    // g_print("safety parent %p  current %p \n", obj_meta->parent, obj_meta);
                    if (obj_meta->parent) {
                        g_print ("Safety found for parent object %p (type=%s)\n",
                          obj_meta->parent,  obj_meta->parent->obj_label);
                    }

              }
            }

            for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next) {
                class_meta = (NvDsClassifierMeta *)(l_class->data);
                if (!class_meta) continue;
                if (class_meta->unique_component_id == SGIE1_CLASSIFIER_UID) {
                    // g_print("class meta uid: %d  num labels is %d\n", class_meta->unique_component_id, class_meta->num_labels);
                    for (lable_i = 0, l_label = class_meta->label_info_list; lable_i < class_meta->num_labels && l_label; lable_i++, l_label = l_label->next) {
                        label_info = (NvDsLabelInfo *) (l_label->data);
                        if (label_info) {
                            if (label_info->label_id == 0 && label_info->result_class_id == 0) {
                                g_print("label id: %d , result class id:%d, result label: %s \n ", label_info->label_id, label_info->result_class_id, label_info->result_label);
                                // g_print("label id: %d , result class id:%d, result label: %s \n ", label_info->label_id, label_info->result_class_id, label_info->result_label);
                            }
                        }
                    }
                }
            }

            // /* Now set the offsets where the string should appear */
            // txt_params->x_offset = obj_meta->rect_params.left;
            // txt_params->y_offset = obj_meta->rect_params.top - 25;

            // /* Font , font-color and font-size */
            // txt_params->font_params.font_name = (char *)("Serif");
            // txt_params->font_params.font_size = 10;
            // txt_params->font_params.font_color.red = 1.0;
            // txt_params->font_params.font_color.green = 1.0;
            // txt_params->font_params.font_color.blue = 1.0;
            // txt_params->font_params.font_color.alpha = 1.0;

            // /* Text background color */
            // txt_params->set_bg_clr = 1;
            // txt_params->text_bg_clr.red = 0.0;
            // txt_params->text_bg_clr.green = 0.0;
            // txt_params->text_bg_clr.blue = 0.0;
            // txt_params->text_bg_clr.alpha = 1.0;

            /*
            * Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
            * component implementing detection / recognition logic.
            * Here it demonstrates how to use / attach that meta data.
            */
            if (is_first_object && !(frame_number % frame_interval) && safety_count) {
              /* Frequency of messages to be send will be based on use case.
              * Here message is being sent for first object every frame_interval(default=30).
              */
              
              /**
               * 事件定义：
               * 未系安全带事件
               *    车头检测到未系安全带，则事件发生
               * 摩托车未戴头盔事件
               *    摩托车区域内检测到未带头盔，则事件发生
               * 
               * 上报策略：
               * 对于相同trackid的对象，可进行事件计数，达到某一阈值则向数据中台上报事件。
               * 
              */
              // save img at local device
              /* To verify  encoded metadata of cropped objects, we iterate through the
              * user metadata of each object and if a metadata of the type
              * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
              * implemented below.
              */
              char fileNameString[FILE_NAME_SIZE];
              const char *osd_string = "OSD";
              const char *save_img_path = "/opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-perf-test-pipeline1/result_imgs/";
              int obj_res_width = (int) obj_meta->rect_params.width;
              int obj_res_height = (int) obj_meta->rect_params.height;
              // if(prop.integrated) {
              //   obj_res_width = GST_ROUND_DOWN_2(obj_res_width);
              //   obj_res_height = GST_ROUND_DOWN_2(obj_res_height);
              // }

              snprintf (fileNameString, FILE_NAME_SIZE, "%s%s_%d_%d_%d_%s_%dx%d.jpg",
                  save_img_path,
                  osd_string, frame_number, frame_meta->source_id, num_rects,
                  obj_meta->obj_label, obj_res_width, obj_res_height);
              g_print("---------------save image buffer to file %s\n", fileNameString);
              /* For Demonstration Purposes we are writing metadata to jpeg images of
              * only vehicles for the first 100 frames only.
              * The files generated have a 'OSD' prefix. */
              if (obj_meta->class_id == SGIE1_CLASS_ID_UNSAFETYBELT && 0) {
                g_print("---------------save image buffer to file %s\n", fileNameString);
                NvDsUserMetaList *usrMetaList = obj_meta->obj_user_meta_list;
                FILE *file;
                while (usrMetaList != NULL) {
                  NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
                  if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
                    NvDsObjEncOutParams *enc_jpeg_image =
                        (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
                    /* Write to File */
                    file = fopen (fileNameString, "wb");
                    fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                        enc_jpeg_image->outLen, file);
                    fclose (file);
                    usrMetaList = NULL;
                  } else {
                    usrMetaList = usrMetaList->next;
                  }
                }
              }

              NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
              msg_meta->bbox.top = obj_meta->rect_params.top;
              msg_meta->bbox.left = obj_meta->rect_params.left;
              msg_meta->bbox.width = obj_meta->rect_params.width;
              msg_meta->bbox.height = obj_meta->rect_params.height;
              msg_meta->frameId = frame_number;
              msg_meta->trackingId = obj_meta->object_id;
              msg_meta->confidence = obj_meta->confidence;


              // {
              //     "messageid": "08826813-2f56-4bc1-9e6b-8c507a81917b",
              //     "mdsversion": "1.0",
              //     "@timestamp": "2022-12-26T01:47:49.389Z",
              //     "analyticsModule": {
              //         "id": "001",
              //         "description": "Traffic Violation Detection",
              //         "source": "OpenALR",
              //         "version": "1.0"
              //     },
              //     "object": {
              //         "id": "18446744073709551615",
              //         "violation": {
              //             "type": "unsafety-belt",
              //             "license": "",
              //             "confidence": 0.724534273147583
              //         },
              //         "bbox": {
              //             "topleftx": 809,
              //             "toplefty": 320,
              //             "bottomrightx": 1279,
              //             "bottomrighty": 732
              //         }
              //     },
              //     "event": {
              //         "id": "bbd352fb-2d67-4384-abc2-e7e943179556",
              //         "type": "unsafety-belt"
              //     },
              //     "videoPath": "",
              //     "imagePath": ""
              // }

              generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta, fileNameString);

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

        //TODO add display at frame
        frame_meta->display_meta_list;
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        // g_print("display meta rect num is %d\n", display_meta->num_rects);
        // NvOSD_RectParams *rect_params = (NvOSD_RectParams*)&display_meta->rect_params[0];
        // rect_params->left = example_obj.left;
        // rect_params->top = example_obj.top;
        // rect_params->width = example_obj.width;
        // rect_params->height = example_obj.height;
        // rect_params->has_bg_color = 0;
        // rect_params->bg_color = (NvOSD_ColorParams) {1, 1, 0, 0.4};
        // rect_params->border_width = 1;
        // rect_params->border_color = (NvOSD_ColorParams) {0, 0, 1, 1};
        // display_meta->num_rects = 1;

        NvOSD_TextParams *text_params = (NvOSD_TextParams*)&display_meta->text_params[0];
        display_meta->num_labels = 1;
        text_params->display_text = (char*) g_malloc0 (MAX_DISPLAY_LEN);
        int offset = 0;
        offset = snprintf(text_params->display_text, MAX_DISPLAY_LEN, "Person = %d", person_count);
        offset = snprintf(text_params->display_text + offset, MAX_DISPLAY_LEN, " Car = %d", vehicle_count);
        text_params->x_offset = 10;
        text_params->y_offset =12;
        text_params->font_params.font_name = (char*) ("Serif");
        text_params->font_params.font_size = 10;
        text_params->font_params.font_color.red = 1.0;
        text_params->font_params.font_color.green = 1.0;
        text_params->font_params.font_color.blue = 1.0;
        text_params->font_params.font_color.alpha = 1.0;
        text_params->set_bg_clr = 1;
        text_params->text_bg_clr.red = 0.0;
        text_params->text_bg_clr.green = 0.0;
        text_params->text_bg_clr.blue = 0.0;
        text_params->text_bg_clr.alpha = 0.5;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);    
        // g_print("display meta rect num is %d\n", display_meta->num_rects);


        frame_number++;
        // g_print("frame num is %d\n", frame_number);
    }
    // g_print ("Frame Number = %d "
    //     "Vehicle Count = %d Person Count = %d Safety Count = %d \n",
    //     frame_number, vehicle_count, person_count, safety_count);

    return GST_PAD_PROBE_OK;
}



static GstPadProbeReturn
sgie2_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint safety_count = 0;
    guint lable_i = 0;
    gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj, *l_class, *l_label;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsClassifierMeta * class_meta = NULL;
    NvDsLabelInfo * label_info = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
        // No batch meta attached.
        return GST_PAD_PROBE_OK;
    }

    // g_print("sgie2 batch frame meta num is %d\n", batch_meta->num_frames_in_batch);
    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;

        // g_print("sgie2 frame meta num is %d\n", frame_meta->num_obj_meta);
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
            
            // NvDsObjectMeta *obj_uu = nvds_acquire_obj_meta_from_pool(batch_meta);
            // obj_uu->class_id = 110;
            // // nvds_add_obj_meta_to_frame(frame_meta, obj_uu, obj_meta);
            // nvds_add_obj_meta_to_frame(frame_meta, obj_uu, obj_meta);
            // // obj_uu->parent = obj_meta;
            // // g_print("parent %p \n", obj_uu->parent);

            for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next) {
                class_meta = (NvDsClassifierMeta *)(l_class->data);
                if (!class_meta) continue;
                if (class_meta->unique_component_id == SGIE1_CLASSIFIER_UID) {
                    // g_print("sgie2 class meta uid: %d  num labels is %d\n", class_meta->unique_component_id, class_meta->num_labels);
                    for (lable_i = 0, l_label = class_meta->label_info_list; lable_i < class_meta->num_labels && l_label; lable_i++, l_label = l_label->next) {
                        label_info = (NvDsLabelInfo *) (l_label->data);
                        if (label_info) {
                            // g_print("sgie2 label id: %d , result class id:%d, result label: %s \n ", label_info->label_id, label_info->result_class_id, label_info->result_label);
                            // g_print("sgie2 class id b %d\n", obj_meta->class_id);
                            // g_print("sgie2 class id %d\n", obj_meta->class_id);
                            if (label_info->label_id == 0) {
                            // if (label_info->label_id == 0 && label_info->result_class_id == SGIE0_CLASS_ID_HEAD) {
                                obj_meta->class_id = label_info->result_class_id;
                                obj_meta->unique_component_id = class_meta->unique_component_id;
                                // g_print("obj meta label %s, conf %f\n", obj_meta->obj_label, obj_meta->confidence);
                                // g_print("label id: %d , result class id:%d, result label: %s \n ", label_info->label_id, label_info->result_class_id, label_info->result_label);
                            }
                        }
                    }
                }
            }
          }
        }

        // g_print("display meta rect num is %d\n", display_meta->num_rects);
    return GST_PAD_PROBE_OK;
}

#define save_img TRUE
// #define save_img FALSE
#define attach_user_meta FALSE
// #define attach_user_meta TRUE
bool save_cropped_images_enabled = true;
bool save_full_frame_enabled = true;

static GstPadProbeReturn
sgie2_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer ctx)
{
    GstBuffer *buf = (GstBuffer *) info->data;

    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
      GST_ERROR ("input buffer mapinfo failed");
      // return GST_FLOW_ERROR;
      return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
    gst_buffer_unmap (buf, &inmap);


    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    NvOSD_RectParams *rect_params = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint safety_count = 0;
    guint lable_i = 0;
    gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj, *l_class, *l_label;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsClassifierMeta * class_meta = NULL;
    NvDsLabelInfo * label_info = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
        // No batch meta attached.
        return GST_PAD_PROBE_OK;
    }

    // g_print("sgie2 batch frame meta num is %d\n", batch_meta->num_frames_in_batch);
    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;

        // g_print("sgie2 frame meta num is %d\n", frame_meta->num_obj_meta);
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
            // if (obj_meta->class_id == 110) {
            //   g_print("110 parnet is %p  %s\n", obj_meta->parent, obj_meta->parent->obj_label);
            // }


            if (obj_meta->unique_component_id == SGIE2_DETECTOR_UID) {


              txt_params = &(obj_meta->text_params);
              // if (txt_params->display_text)
              //   g_free (txt_params->display_text);

              // txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);

              // g_snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "%s ",
              //             pgie_classes_str[obj_meta->class_id]);


              /* Now set the offsets where the string should appear */
              // txt_params->x_offset = obj_meta->rect_params.left;
              // txt_params->y_offset = obj_meta->rect_params.top - 25;

              /* Font , font-color and font-size */
              txt_params->font_params.font_name = "Serif";
              txt_params->font_params.font_size = 5;
              txt_params->font_params.font_color.red = 1.0;
              txt_params->font_params.font_color.green = 1.0;
              txt_params->font_params.font_color.blue = 1.0;
              txt_params->font_params.font_color.alpha = 1.0;

              /* Text background color */
              txt_params->set_bg_clr = 1;
              txt_params->text_bg_clr.red = 0.1;
              txt_params->text_bg_clr.green = 0.3;
              txt_params->text_bg_clr.blue = 0.4;
              txt_params->text_bg_clr.alpha = 1.0;

              rect_params = &(obj_meta->rect_params);
              rect_params->border_width = 1;

              // g_print("safety detect result is %s(%d)\n", obj_meta->obj_label, obj_meta->class_id);
              if (obj_meta->class_id == SGIE1_CLASS_ID_UNSAFETYBELT) {
                  safety_count++;
                  
                  if (is_first_object && !(frame_number % frame_interval) && safety_count) {
                      /* Frequency of messages to be send will be based on use case.
                      * Here message is being sent for first object every frame_interval(default=30).
                      */

                      /**
                      * 事件定义：
                      * 未系安全带事件
                      *    车头检测到未系安全带，则事件发生
                      * 摩托车未戴头盔事件
                      *    摩托车区域内检测到未带头盔，则事件发生
                      * 
                      * 上报策略：
                      * 对于相同trackid的对象，可进行事件计数，达到某一阈值则向数据中台上报事件。
                      * 
                      */
                      // save img at local device
                      /* To verify  encoded metadata of cropped objects, we iterate through the
                      * user metadata of each object and if a metadata of the type
                      * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
                      * implemented below.
                      */
                      char fileNameString[FILE_NAME_SIZE];
                      const char *osd_string = "OSD";
                      const char *save_img_path = "/opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-perf-test-pipeline1/result_imgs/";
                      int obj_res_width = (int) obj_meta->rect_params.width;
                      int obj_res_height = (int) obj_meta->rect_params.height;
                      // if(prop.integrated) {
                      //   obj_res_width = GST_ROUND_DOWN_2(obj_res_width);
                      //   obj_res_height = GST_ROUND_DOWN_2(obj_res_height);
                      // }

                      char fileNameStringFull[FILE_NAME_SIZE];
                      const char *save_full_img_path = "/opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-perf-test-pipeline1/full_result_imgs/";
                      snprintf (fileNameStringFull, FILE_NAME_SIZE, "%s%s_%d_%d_%d_%s_%dx%d.jpg",
                          save_full_img_path,
                          osd_string, frame_number, frame_meta->source_id, 1,
                          obj_meta->obj_label, ip_surf->surfaceList[frame_meta->batch_id].width, ip_surf->surfaceList[frame_meta->batch_id].height);


                      snprintf (fileNameString, FILE_NAME_SIZE, "%s%s_%d_%d_%d_%s_%dx%d.jpg",
                          save_img_path,
                          osd_string, frame_number, frame_meta->source_id, 1,
                          obj_meta->obj_label, obj_res_width, obj_res_height);
                      // g_print("---------------save image buffer to file %s\n", fileNameString);

                      NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
                      msg_meta->bbox.top = obj_meta->rect_params.top;
                      msg_meta->bbox.left = obj_meta->rect_params.left;
                      msg_meta->bbox.width = obj_meta->rect_params.width;
                      msg_meta->bbox.height = obj_meta->rect_params.height;
                      msg_meta->frameId = frame_number;
                      msg_meta->trackingId = obj_meta->object_id;
                      msg_meta->confidence = obj_meta->confidence;
                      msg_meta->otherAttrs = g_strdup(fileNameStringFull);
                      // g_print("------full image path %s\n", msg_meta->otherAttrs);


                      // {
                      //     "messageid": "08826813-2f56-4bc1-9e6b-8c507a81917b",
                      //     "mdsversion": "1.0",
                      //     "@timestamp": "2022-12-26T01:47:49.389Z",
                      //     "analyticsModule": {
                      //         "id": "001",
                      //         "description": "Traffic Violation Detection",
                      //         "source": "OpenALR",
                      //         "version": "1.0"
                      //     },
                      //     "object": {
                      //         "id": "18446744073709551615",
                      //         "violation": {
                      //             "type": "unsafety-belt",
                      //             "license": "",
                      //             "confidence": 0.724534273147583
                      //         },
                      //         "bbox": {
                      //             "topleftx": 809,
                      //             "toplefty": 320,
                      //             "bottomrightx": 1279,
                      //             "bottomrighty": 732
                      //         },
                      //         "imagePath": ""
                      //     },
                      //     "event": {
                      //         "id": "bbd352fb-2d67-4384-abc2-e7e943179556",
                      //         "type": "unsafety-belt"
                      //     },
                      //     "videoPath": "",
                      //     "fullImagePath": ""
                      // }

                      generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta, fileNameString);

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
                      // is_first_object = FALSE;

                      // enc image to buffer and save at local device 
                      NvDsObjEncUsrArgs userData = { 0 };
                      /* To be set by user */
                      userData.saveImg = save_img;
                      userData.attachUsrMeta = attach_user_meta;
                      /* Set if Image scaling Required */
                      userData.scaleImg = FALSE;
                      userData.scaledWidth = 0;
                      userData.scaledHeight = 0;
                      /* Preset */
                      userData.objNum = 1;
                      /* Quality */
                      userData.quality = 80;
                      // userData.fileNameImg = &fileNameString;
                      memcpy(userData.fileNameImg, fileNameString, sizeof(fileNameString));
                      // g_print("file name is %s\n", userData.fileNameImg);

                      if (save_cropped_images_enabled) {
                          /* Conditions that user needs to set to encode the detected objects of
                            * interest. Here, by default all the detected objects are encoded.
                            * For demonstration, we will encode the first object in the frame */
                          /*Main Function Call */
                          if (obj_meta->parent && 0) {
                              g_print ("Safety found for parent object %p (type=%s)\n",
                                obj_meta->parent,  obj_meta->parent->obj_label);
                              nvds_obj_enc_process ((NvDsObjEncCtxHandle)ctx, &userData, ip_surf, obj_meta->parent, frame_meta);
                          } else {
                              nvds_obj_enc_process ((NvDsObjEncCtxHandle)ctx, &userData, ip_surf, obj_meta, frame_meta);
                          }

                      }
                      if (save_full_frame_enabled) {
                        if (is_first_object) {
                            memcpy(userData.fileNameImg, fileNameStringFull, sizeof(fileNameStringFull));
                            unsigned dummy_counter = 0;
                            /// Creating a special object meta in order to save a full frame
                            NvDsObjectMeta dummy_obj_meta;
                            dummy_obj_meta.rect_params.width = ip_surf->surfaceList[frame_meta->batch_id].width;
                            dummy_obj_meta.rect_params.height = ip_surf->surfaceList[frame_meta->batch_id].height;
                            dummy_obj_meta.rect_params.top = 0;
                            dummy_obj_meta.rect_params.left = 0;
                            nvds_obj_enc_process((NvDsObjEncCtxHandle)ctx, &userData, ip_surf, &dummy_obj_meta, frame_meta);
                        }
                      }
                      is_first_object = FALSE;

                      // // save buffer to local file at userMeta info
                      // NvDsUserMetaList *usrMetaList = obj_meta->obj_user_meta_list;
                      // FILE *file;
                      // while (usrMetaList != NULL) {
                      //   NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
                      //   if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
                      //     NvDsObjEncOutParams *enc_jpeg_image =
                      //         (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
                      //     /* Write to File */
                      //     file = fopen (fileNameString, "wb");
                      //     fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                      //         enc_jpeg_image->outLen, file);
                      //     fclose (file);
                      //     usrMetaList = NULL;
                      //   } else {
                      //     usrMetaList = usrMetaList->next;
                      //   }
                      // }
                  }
                    
                    

              }
            }
        }
        frame_number++;
    }

    // g_print("display meta rect num is %d\n", display_meta->num_rects);
    nvds_obj_enc_finish((NvDsObjEncCtxHandle)ctx);
    return GST_PAD_PROBE_OK;
}

int sink_id = 0;
/* This function will be called by the pad-added signal */
static void pad_added_handler (GstElement *src, GstPad *new_pad, GstElement *data) {
  
  char sink_str[9] ;
  snprintf(sink_str, 9, "sink_%d", sink_id);
  GstPad *sink_pad = gst_element_get_request_pad (data, sink_str);
  GstPadLinkReturn ret;
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;

  if (!sink_pad) {
      g_printerr ("Streammux request sink pad failed at pad added handler. Exiting.\n");
      g_print ("sinkpad Obtained request pad %s.\n", gst_pad_get_name (sink_pad));
      goto exit;
  }

  g_print ("Received data '%s' from '%s':\n", GST_PAD_NAME (sink_pad), GST_ELEMENT_NAME (data));
  g_print ("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

  /* If our converter is already linked, we have nothing to do here */
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("We are already linked. Ignoring.\n");
    goto exit;
  }

  /* Check the new pad's type */
  new_pad_caps = gst_pad_get_current_caps (new_pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);
  if (!g_str_has_prefix (new_pad_type, "video/x-raw")) {
    g_print ("It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
    goto exit;
  }

  /* Attempt the link */


  // if (gst_pad_link (new_pad, sink_pad) != GST_PAD_LINK_OK) {
  //     g_printerr ("Failed to link decoder to stream muxer in pad added handler. Exiting.\n");
  //     // return -1;
  // }

  ret = gst_pad_link (new_pad, sink_pad);
  // g_print("sssssssss\n");



  if (GST_PAD_LINK_FAILED (ret)) {
    g_print ("Type is '%s' but link failed.\n", new_pad_type);
  } else {
    g_print ("Link succeeded (type '%s').\n", new_pad_type);
    // sink_id++;
  }

exit:
  /* Unreference the new pad's caps, if we got them */
  if (new_pad_caps != NULL)
    gst_caps_unref (new_pad_caps);

  /* Unreference the sink pad */
  gst_object_unref (sink_pad);
}

int main(int argc, char* argv[]) {

    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *streammux = NULL,
                *decoder = NULL, *sink = NULL, *nvvidconv = NULL, *nvosd = NULL,
                *pgie = NULL, *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL, 
                *nvvidconv2 = NULL, *nvtracker1 = NULL;
    GstElement *tee = NULL, *msgconv = NULL, *msgbroker = NULL,
               *queue1 = NULL, *queue2 = NULL;
    GstPad *osd_sink_pad = NULL, *tee_render_pad = NULL, *tee_msg_pad = NULL;
    GstElement *transform = NULL;
    GstBus* bus = NULL;
    guint bus_watch_id;
    GstPad *dec_src_pad = NULL;
    GstPad *sgie2_sink_pad = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if (argc != 4) {
        g_printerr("Usage: %s <rows num> <columns num> <streams dir>\n", argv[0]);
        return -1;
    }

    get_file_list(argv[3]);

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("perf-test-pipeline1");

    /* Source element for reading from the file */
    source = gst_element_factory_make("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make("h264parse", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    // decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
    decoder = gst_element_factory_make ("decodebin", "decodebin-decoder");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* We need three secondary gies so lets create 3 more instances of
       nvinfer */
    sgie1 = gst_element_factory_make ("nvinfer", "secondary1-nvinference-engine");
    sgie2 = gst_element_factory_make ("nvinfer", "secondary2-nvinference-engine");
    nvtracker1 = gst_element_factory_make ("nvtracker", "nvtracker1");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");

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
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  } else {
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    // sink = gst_element_factory_make ("fpsdisplaysink", "nvvideo-renderer");
  } 

    /* caps filter for nvvidconv to convert NV12 to RGBA as nvosd expects input
     * in RGBA format */
    if (!source || !h264parser || !decoder || !nvvidconv || !pgie
            || !sgie1 || !sgie2 || !nvosd || !sink || !nvvidconv2 || !nvtracker1
            || !msgconv || !msgbroker || !tee  || !queue1 || !queue2) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

  if(!transform && prop.integrated) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
  for (int i = 0; i < file_list.size(); i++) {
    g_print("file list %s\n", file_list[i].c_str());
  }

    /* Set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", file_list[0].c_str(), NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT, "batch-size", 1,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    // g_object_set (G_OBJECT(decoder), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(nvvidconv), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(nvvidconv2), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(nvosd), "gpu-id", GPU_ID, NULL);

    /* we set the osd properties here */
    g_object_set(G_OBJECT(nvosd), "clock-font-size", 12, 
                                  "clock-color", "1;0;0;0",
                                //   "text-bg-color", "0.3;0.3;0.3;1;",
                                  "clock-font", "Serif",
                                  "x-clock-offset", 800,
                                  "y-clock-offset", 800,
                                  NULL);
    // g_object_set(G_OBJECT(nvosd), "font-size", 15, NULL);
    // g_object_set(G_OBJECT(nvosd), "font-size", 15, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", SGIE1_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT (sgie2), "config-file-path", SGIE2_CONFIG_FILE, NULL);

    g_object_set (G_OBJECT(pgie), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(sgie1), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(sgie2), "gpu-id", GPU_ID, NULL);

    // g_object_set (G_OBJECT(pgie), "output-tensor-meta", TRUE, NULL);

    // g_object_set (G_OBJECT(nvtracker1), "ll-lib", "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so", NULL);

    g_object_set (G_OBJECT (nvtracker1),
      "ll-lib-file", "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so",
      "ll-config-file", "../../../../samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
      "tracker-width", 640, "tracker-height", 480,
       NULL);



    // ./deepstream-test-upload  -i ../../../../samples/streams/s/sample_720p.h264 
    //  -p ../../../../lib/libnvds_amqp_proto.so 
    // --conn-str "192.168.3.18;5672;guest;guest" -s 0 --msg2p-meta 1

    g_object_set (G_OBJECT(msgconv), "config", MSCONV_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT(msgconv), "payload-type", schema_type, NULL);
    g_object_set (G_OBJECT(msgconv), "msg2p-newapi", msg2p_meta, NULL);
    g_object_set (G_OBJECT(msgconv), "frame-interval", frame_interval, NULL);

    g_object_set (G_OBJECT(msgbroker), "proto-lib", proto_lib,
                    "conn-str", conn_str, "sync", FALSE, NULL);

    if (topic) {
        g_object_set (G_OBJECT(msgbroker), "topic", topic, NULL);
    }

    if (cfg_file) {
        g_object_set (G_OBJECT(msgbroker), "config", cfg_file, NULL);
    }


    // ll-lib-file=/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so 

    // g_object_set (G_OBJECT (sink), "sync", FALSE, "max-lateness", -1,
    //               "async", FALSE, NULL);
    g_object_set (G_OBJECT(sink), "gpu-id", GPU_ID, NULL);
    g_object_set (G_OBJECT(sink), "rows", atoi(argv[1]), NULL);
    g_object_set (G_OBJECT(sink), "columns", atoi(argv[2]), NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    gst_bin_add_many (GST_BIN (pipeline),
        // source, h264parser, decoder, streammux, pgie, nvtracker1, sgie1, sgie2, nvvidconv,
        source,  decoder, streammux, pgie, nvtracker1, sgie1, sgie2, nvvidconv,
        // source, h264parser, decoder, streammux, pgie, nvtracker1, sgie1, sgie2, 
        nvosd, sink, tee, queue1, queue2, msgconv, msgbroker, NULL);
    if(prop.integrated) {
        gst_bin_add (GST_BIN (pipeline), transform);
    }
    else {
        gst_bin_add (GST_BIN (pipeline), nvvidconv2);
    }
    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder -> nvstreammux ->
    * nvinfer -> nvtracker -> nvinfer -> nvinfer 
    *         -> nvvidconv -> nvosd -> tee -> queue -> video-renderer
    *                                     |
    *                                     |-> queue -> msgconv -> msgbroker  */

    g_signal_connect (decoder, "pad-added", G_CALLBACK (pad_added_handler), streammux);


    // GstPad *sinkpad, *srcpad;
    // gchar pad_name_sink[16] = "sink_0";
    // gchar pad_name_src[16] = "src_0";

    // sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    // g_print ("sinkpad Obtained request pad %s.\n", gst_pad_get_name (sinkpad));
    // if (!sinkpad) {
    //     g_printerr ("Streammux request sink pad failed. Exiting.\n");
    //     return -1;
    // }

    // // srcpad = gst_element_get_static_pad (decoder, "src");
    // srcpad = gst_element_get_request_pad (decoder, pad_name_src);
    // g_print ("srcpad Obtained request pad %s.\n", gst_pad_get_name (srcpad));
    // // srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    // if (!srcpad) {
    //     g_printerr ("Decoder request src pad failed. Exiting.\n");
    //     return -1;
    // }
    // gst_element_link_pads (source, "src", decoder, "sink");

    // if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    //     g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
    //     return -1;
    // }

    // gst_object_unref (sinkpad);
    // gst_object_unref (srcpad);

    /* Link the elements together */
    if (!gst_element_link_many (source, decoder, NULL)) {
    // if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    // if (!gst_element_link_many (streammux, pgie, sgie1, nvtracker1,
    if (!gst_element_link_many (streammux, pgie, nvtracker1, sgie1,
                                sgie2, nvvidconv, nvosd, tee, NULL)) {
                                // sgie2,  nvosd, tee, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
    }

    if (!gst_element_link_many (queue1, msgconv, msgbroker, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
    }

    if(prop.integrated) {
        if (!gst_element_link_many (queue2, transform, sink, NULL)) {
            g_printerr ("Elements could not be linked. Exiting.\n");
            return -1;
        }
    }
    else {
        if (!gst_element_link_many (queue2, nvvidconv2, sink, NULL)) {
            g_printerr ("Elements could not be linked. Exiting.\n");
            return -1;
        }
    }

    GstPad *sink_pad;
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


    // dec_src_pad = gst_element_get_static_pad(decoder, "sink");
    // if (!dec_src_pad) {
    //     g_print("Unable to get h264parser src pad \n");
    // } else {
    //     g_print("able to get h264parser src pad \n");
    //     gst_pad_add_probe(dec_src_pad,
    //                       (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
    //                                          GST_PAD_PROBE_TYPE_EVENT_FLUSH |
    //                                          GST_PAD_PROBE_TYPE_BUFFER),
    //                       eos_probe_cb, pipeline, NULL);
    //     gst_object_unref(dec_src_pad);
    // }


    /* Lets add probe to get informed of the meta data generated, we add probe to
    * the sink pad of the osd element, since by that time, the buffer would have
    * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else {
        if(msg2p_meta == 0) //generate payload using eventMsgMeta
            gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                osd_sink_pad_buffer_probe, NULL, NULL);
        }
    gst_object_unref (osd_sink_pad);

    /*Creat Context for Object Encoding */
    NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context ();

    osd_sink_pad = gst_element_get_static_pad (nvosd, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get src pad\n");
    else {
        if(msg2p_meta == 0) //generate payload using eventMsgMeta
            gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                osd_src_pad_buffer_probe, obj_ctx_handle, NULL);
        }
    gst_object_unref (osd_sink_pad);


    if (!obj_ctx_handle) {
      g_print ("Unable to create context\n");
      return -1;
    }

    sgie2_sink_pad = gst_element_get_static_pad (sgie2, "sink");
    if (!sgie2_sink_pad) {
        g_print("Unable to get sgie2 sink pad\n");
    } else {
      
      gst_pad_add_probe(sgie2_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          sgie2_sink_pad_buffer_probe, obj_ctx_handle, NULL);
    }
    gst_object_unref(sgie2_sink_pad);

    sgie2_sink_pad = gst_element_get_static_pad (sgie2, "src");
    if (!sgie2_sink_pad) {
        g_print("Unable to get sgie2 src pad\n");
    } else {
      gst_pad_add_probe(sgie2_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          sgie2_src_pad_buffer_probe, obj_ctx_handle, NULL);
    }
    gst_object_unref(sgie2_sink_pad);
    


    /* Set the pipeline to "playing" state */
    g_print("Now playing: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Destroy context for Object Encoding */
    nvds_obj_enc_destroy_context (obj_ctx_handle);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
