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

#include "deepstream_action.h"

/** Defines the maximum size of a string. */
#define MAX_STR_LEN 1024

/** Defines the maximum size of an array for storing a text result. */
#define MAX_LABEL_SIZE 128

/** 3D model input NCDHW has 5 dims; 2D model input NSHW has 4 dims */
#define MODEL_3D_SHAPES 5

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* Action recognition config */
static NvDsARConfig gActionConfig;

/* Check signal interrupt invoked */
static volatile gboolean gIntr = false;

/* main gstreamer pipeline */
static volatile GstElement *gPipeline = nullptr;

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Debug envrionment variable name for libnvds_custom_sequence_preprocess.so */
#define ENV_CUSTOM_SEQUENC_DEBUG "DS_CUSTOM_SEQUENC_DEBUG"

#define MAX_CLASS_LEN 5
static const gchar kActioClasseLabels[MAX_CLASS_LEN][MAX_LABEL_SIZE] = {
    "push", "fall_floor" , "walk", "run", "ride_bike"};

/* add fps display metadata into frame */
static void
add_fps_display_meta(NvDsFrameMeta *frame, NvDsBatchMeta *batch_meta) {
  static FpsCalculation fpsCal(50);
  float fps = fpsCal.updateFps(frame->source_id);
  if (fps < 0) {
    return;
  }

  NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
  display_meta->num_labels = 1;
  NvOSD_TextParams *txt_params = &display_meta->text_params[0];
  txt_params->display_text = (char *)g_malloc0(MAX_STR_LEN);

  snprintf(txt_params->display_text, MAX_STR_LEN - 1, "FPS: %.2f", fps);
  /* Now set the offsets where the string should appear */
  txt_params->x_offset = 0;
  txt_params->y_offset = 40;

  /* Font , font-color and font-size */
  txt_params->font_params.font_name = (char *)"Serif";
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

  nvds_add_display_meta_to_frame(frame, display_meta);
}

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  NvDsMetaList *l_user_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  for (l_user_meta = batch_meta->batch_user_meta_list; l_user_meta != NULL;
       l_user_meta = l_user_meta->next)
  {
    user_meta = (NvDsUserMeta *)(l_user_meta->data);
    if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META)
    {
      GstNvDsPreProcessBatchMeta *preprocess_batchmeta =
          (GstNvDsPreProcessBatchMeta *)(user_meta->user_meta_data);
      std::string model_dims = "";
      if (preprocess_batchmeta->tensor_meta) {
        if (preprocess_batchmeta->tensor_meta->tensor_shape.size() == MODEL_3D_SHAPES) {
          model_dims = "3D: AR - ";
        } else {
          model_dims = "2D: AR - ";
        }
      }
      for (auto &roi_meta : preprocess_batchmeta->roi_vector)
      {
        NvDsMetaList *l_user = NULL;
        for (l_user = roi_meta.roi_user_meta_list; l_user != NULL;
             l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
          if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
          {
            NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)(user_meta->user_meta_data);
            gfloat max_prob = 0;
            gint class_id = -1;
            gfloat *buffer = (gfloat *)tensor_meta->out_buf_ptrs_host[0];
            for (size_t i = 0; i < tensor_meta->output_layers_info[0].inferDims.d[0]; i++)
            {
              if (buffer[i] > max_prob)
              {
                max_prob = buffer[i];
                class_id = i;
              }
            }
            const gchar *label = "";
            if (class_id < MAX_CLASS_LEN)
              label = kActioClasseLabels[class_id];
            LOG_DEBUG("output tensor result: cls_id: %d, scrore:%.3f, label: %s", class_id, max_prob, label);
          }
        }

        NvDsMetaList *l_classifier = NULL;
        for (l_classifier = roi_meta.classifier_meta_list; l_classifier != NULL;
             l_classifier = l_classifier->next)
        {
          NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)(l_classifier->data);
          NvDsLabelInfoList *l_label;
          for (l_label = classifier_meta->label_info_list; l_label != NULL;
               l_label = l_classifier->next)
          {
            NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;

            NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            display_meta->num_labels = 1;

            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            txt_params->display_text = (char *)g_malloc0(MAX_STR_LEN);

            snprintf(txt_params->display_text, MAX_STR_LEN - 1,
                     "%s: %s", model_dims.c_str(), label_info->result_label);
            LOG_DEBUG("classification result: cls_id: %d, label: %s", label_info->result_class_id, label_info->result_label);
            /* Now set the offsets where the string should appear */
            txt_params->x_offset = roi_meta.roi.left;
            txt_params->y_offset = (uint32_t)std::max<int32_t>(roi_meta.roi.top - 10, 0);

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = (char *)"Serif";
            txt_params->font_params.font_size = 12;
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

            nvds_add_display_meta_to_frame(roi_meta.frame_meta, display_meta);
          }
        }
      }
    }
  }

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    // print FPS on each stream
    if (gActionConfig.enableFps) {
      add_fps_display_meta(frame_meta, batch_meta);
    }
  }

  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_WARNING:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_warning(msg, &error, &debug);
    g_printerr("WARNING from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    g_free(debug);
    g_printerr("Warning: %s\n", error->message);
    g_error_free(error);
    break;
  }
  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
#ifndef PLATFORM_TEGRA
  case GST_MESSAGE_ELEMENT:
  {
    if (gst_nvmessage_is_stream_eos(msg))
    {
      guint stream_id;
      if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
      {
        g_print("Got EOS from stream %d\n", stream_id);
      }
    }
    break;
  }
#endif
  default:
    break;
  }
  return TRUE;
}

static void
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
  g_print("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstElement *source_bin = (GstElement *)data;
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp(name, "video", 5))
  {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM))
    {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
      if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                    decoder_src_pad))
      {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    }
    else
    {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data)
{
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name)
  {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin(guint index, const gchar *uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin)
  {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                   G_CALLBACK(cb_newpad), bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                            GST_PAD_SRC)))
  {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void _intr_handler (int signum)
{
  gIntr = TRUE;
  g_printerr ("User Interrupted.. \n");

  if (gPipeline) {
    /* Send EOS to the pipeline */
    if (!gst_element_send_event (GST_ELEMENT(gPipeline),
          gst_event_new_eos())) {
      g_print("Interrupted, EOS not sent");
    }
  }
}

/*
* Function to install custom handler for program interrupt signal.
*/
static void _intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
             *preprocess = NULL, *queue1, *queue2, *queue3, *queue4, *queue5, *queue6,
             *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *pgie_src_pad = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);

    /* setup signal handler */
  _intr_setup();

    /* Check input arguments */
  if (argc < 3 || strncmp(argv[1], "-c", 3))
  {
    g_printerr("Usage: %s -c <action_recognition_config.txt>\n", argv[0]);
    return -1;
  }

  if (!parse_action_config(argv[2], gActionConfig)) {
    g_printerr("parse config file: %s failed.\n", argv[2]);
    return -1;
  }

  if (gActionConfig.debug >= kDebugVerbose) {
    setenv(ENV_CUSTOM_SEQUENC_DEBUG, "1", 1);
  } else {
    unsetenv(ENV_CUSTOM_SEQUENC_DEBUG);
  }

  num_sources = gActionConfig.uri_list.size();

  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("preprocess-test-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  for (i = 0; i < num_sources; i++)
  {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = {};
    GstElement *source_bin = create_source_bin(i, gActionConfig.uri_list[i].c_str());

    if (!source_bin)
    {
      g_printerr("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    g_snprintf(pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad(streammux, pad_name);
    if (!sinkpad)
    {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad)
    {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  /* to preprocess the rois and form a raw tensor for inferencing */
  preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make("queue", "queue1");
  queue2 = gst_element_factory_make("queue", "queue2");
  queue3 = gst_element_factory_make("queue", "queue3");
  queue4 = gst_element_factory_make("queue", "queue4");
  queue5 = gst_element_factory_make("queue", "queue5");
  queue6 = gst_element_factory_make("queue", "queue6");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if (prop.integrated)
  {
    transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
  }
  sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

  if (!preprocess || !pgie || !tiler || !nvvidconv || !nvosd || !sink)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!transform && prop.integrated)
  {
    g_printerr("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);

  g_object_set(G_OBJECT(streammux), "width", gActionConfig.muxer_width, "height",
               gActionConfig.muxer_height,
               "batched-push-timeout", gActionConfig.muxer_batch_timeout, NULL);

  g_object_set(G_OBJECT(preprocess), "config-file", gActionConfig.preprocess_config.c_str(), NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set(G_OBJECT(pgie), "input-tensor-meta", TRUE,
               "config-file-path", gActionConfig.infer_config.c_str(), NULL);

  g_print("num-sources = %d\n", num_sources);

  tiler_rows = (guint)sqrt(num_sources);
  tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
               "width", gActionConfig.tiler_width, "height", gActionConfig.tiler_height, NULL);

  g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
               "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set(G_OBJECT(sink), "qos", 0, "sync", gActionConfig.display_sync, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if (prop.integrated)
  {
    gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler, queue4,
                     nvvidconv, queue5, nvosd, queue6, transform, sink, NULL);
    /* we link the elements together
    * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
    if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler, queue4,
                               nvvidconv, queue5, nvosd, queue6, transform, sink, NULL))
    {
      g_printerr("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }
  else
  {
    gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler,
                     queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL);
    /* we link the elements together
    * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
    if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler,
                               queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL))
    {
      g_printerr("Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  pgie_src_pad = gst_element_get_static_pad(pgie, "src");
  if (!pgie_src_pad)
    g_print("Unable to get pgie src pad\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(pgie_src_pad);

  /* Set the pipeline to "playing" state */
  g_print("Now playing:");
  for (i = 0; i < num_sources; i++)
  {
    g_print(" %s,", gActionConfig.uri_list[i].c_str());
  }
  g_print("\n");

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  gPipeline = pipeline;

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  gPipeline = nullptr;

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
