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

#include "deepstream_asr_app.h"

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

  if (i == appctx->num_sources) {
    g_main_loop_quit (loop);
    return FALSE;
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

  ctx = g_option_context_new ("DeepStream-ASR-App");

  g_option_context_add_main_entries (ctx, options, NULL);
  g_option_context_add_group (ctx, gst_init_get_option_group ());
  if (!g_option_context_parse (ctx, &argc, &argv, &err)) {
    g_print ("Error initializing: %s\n", GST_STR_NULL (err->message));
    return 1;
  }

  if (config_file == NULL) {
    g_print ("Application Usage: deepstream_asr_app -c <config file name>\n");
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

  ret = parse_config_file (appctx, config_file);

  if (ret) {
    g_print ("Error in parsing config file %s\n", config_file);
    goto done;
  }

  /* For each user create ASR pipeline */
  int i = 0;

  for (i = 0; i < appctx->num_sources; i++) {
    /* Create audio pipeline */
    ret = create_pipeline (appctx, i, &appctx->sctx[i]);

    if (ret != 0) {
      continue;
    }

    if (start_pipeline (i, &appctx->sctx[i]) != 0) {
      destroy_pipeline (&appctx->sctx[i]);
    }
  }

  g_timeout_add_seconds (2, eos_check_func, appctx);

  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");

  for (i = 0; i < appctx->num_sources; i++) {
    destroy_pipeline (&appctx->sctx[i]);
  }

done:
  g_free (appctx->sctx);
  g_free (appctx);
  g_main_loop_unref (loop);
  return 0;
}
