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


#include "deepstream_asr_tts_app.h"

GMainLoop *loop = NULL;

static gboolean
eos_check_func (gpointer arg)
{
  AppCtx *appctx = (AppCtx *) arg;
  StreamCtx *sctx = NULL;

  guint i;
  gboolean ret = TRUE;

  /* Check if all streams have received eos */
  for (i = 0; i < appctx->num_sources; i++) {
    sctx = &appctx->sctx[i];
    if (sctx->has_audio) {
      if (!sctx->eos_received) {
        break;
      }
    }
  }

  /* Check for EOS on ASR, TTS pipelines or on renderer pipeline if
   * playback is enabled. */
  if ((!appctx->enable_playback && (i == appctx->num_sources)) ||
      (appctx->enable_playback && appctx->eos_received) ) {
    g_main_loop_quit (loop);
    return FALSE;
  }
  return ret;
}

static gboolean
print_position_func (gpointer arg)
{
  AppCtx *appctx = (AppCtx *) arg;
  StreamCtx *sctx = NULL;
  gint64 pos, len;

  guint i;
  gboolean ret = TRUE;

  /* Print stream positions */
  for (i = 0; i < appctx->num_sources; i++) {
    sctx = &appctx->sctx[i];
    if (sctx->has_audio && !sctx->eos_received) {
      if (gst_element_query_position (sctx->asr_pipeline, GST_FORMAT_TIME, &pos)
          && gst_element_query_duration (sctx->asr_pipeline, GST_FORMAT_TIME, &len)) {
        g_print ("Stream %d: Time: %" GST_TIME_FORMAT " / %" GST_TIME_FORMAT "\n",
            i, GST_TIME_ARGS (pos), GST_TIME_ARGS (len));
      }
    }
  }
  if (appctx->enable_playback && !appctx->eos_received) {
    if (gst_element_query_position (appctx->renderer_pipeline, GST_FORMAT_TIME, &pos)) {
          g_print ("Rendering: Time: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS (pos));
    }
  }

  return ret;
}
int
main (int argc, char *argv[])
{
  GOptionContext *ctx = NULL;
  gchar *config_file = NULL;
  GError *err = NULL;
  int ret = 0;

  GOptionEntry options[] = {
    {"config", 'c', 0, G_OPTION_ARG_STRING, &config_file, "config file",
        "file"},
    {NULL}
  };

  ctx = g_option_context_new ("DeepStream-ASR-TTS-App");

  g_option_context_add_main_entries (ctx, options, NULL);
  g_option_context_add_group (ctx, gst_init_get_option_group ());
  if (!g_option_context_parse (ctx, &argc, &argv, &err)) {
    g_print ("Error initializing: %s\n", GST_STR_NULL (err->message));
    return 1;
  }

  if (config_file == NULL) {
    g_print ("Application Usage: deepstream_asr_tts_app -c <config file name>\n");
    return 1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Initialise application context */
  AppCtx *appctx = (AppCtx *) g_malloc0 (sizeof (AppCtx));
  CHECK_PTR (appctx);

  /* Find number of users */
  appctx->num_sources = get_num_sources (config_file);
  g_print ("Number of Input Sources = %d\n", appctx->num_sources);

  appctx->sctx =
      (StreamCtx *) g_malloc0 (sizeof (StreamCtx) * appctx->num_sources);
  CHECK_PTR (appctx->sctx);

  /* Allocate memories for proxy source and sink elements connecting the
   * ASR pipelines with the renderer pipeline. */
  appctx->proxy_audio_sinks =
      (GstElement **) g_malloc0 (sizeof (GstElement *) * appctx->num_sources);
  CHECK_PTR (appctx->proxy_audio_sinks);

  appctx->proxy_audio_sources =
      (GstElement **) g_malloc0 (sizeof (GstElement *) * appctx->num_sources);
  CHECK_PTR (appctx->proxy_audio_sources);


  /* Parse configuration options */
  ret = parse_config_file (appctx, config_file);

  if (ret) {
    g_print ("Error in parsing config file %s\n", config_file);
    goto done;
  }

  /* For each user create ASR pipeline */
  int i = 0;

  for (i = 0; i < appctx->num_sources; i++) {
    /* Create ASR pipeline */
    ret = create_pipeline (appctx, i, &appctx->sctx[i],
        &appctx->proxy_audio_sinks[i]);

    if (ret != 0) {
      return -1;
    }
  }

  if(appctx->enable_playback) {

    /* Create a common pipline for rendering audio and video */
    ret = create_renderer_pipeline (appctx);

    if (ret != 0) {
      return ret;
    }

    /* Set common clock for all pipelines */
    GstClock *clock;

    clock = gst_system_clock_obtain ();
    gst_element_set_base_time (appctx->renderer_pipeline, 0);
    gst_pipeline_use_clock (GST_PIPELINE (appctx->renderer_pipeline), clock);

    for (i = 0; i < appctx->num_sources; i++) {
      gst_pipeline_use_clock (GST_PIPELINE (appctx->sctx[i].asr_pipeline), clock);
      gst_element_set_base_time (appctx->sctx[i].asr_pipeline, 0);
    }
    g_object_unref (clock);
  }

  /* Start the pipelines */
  for (i = 0; i < appctx->num_sources; i++) {
    if (start_pipeline (i, &appctx->sctx[i]) != 0) {
      destroy_pipeline (&appctx->sctx[i]);
    }
  }

  if(appctx->enable_playback) {
    if (gst_element_set_state (appctx->renderer_pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE) {
      return -1;
    }
  }

  g_timeout_add_seconds (2, eos_check_func, appctx);
  g_timeout_add_seconds (2, print_position_func, appctx);

  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");

  for (i = 0; i < appctx->num_sources; i++) {
    destroy_pipeline (&appctx->sctx[i]);
  }

  if(appctx->enable_playback) {
    gst_element_set_state (appctx->renderer_pipeline, GST_STATE_NULL);
  }

done:
  g_free (appctx->sctx);
  g_free (appctx);
  g_main_loop_unref (loop);
  return 0;
}
