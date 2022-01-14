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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CONFIG_GROUP_SOURCE "source"
#define CONFIG_GROUP_SINK "sink"

#define CHECK_PARSE_ERROR(error) \
    if (error) { \
        g_print ("%s", error->message); \
        return 1; \
    }

guint get_num_sources (gchar *config_file)
{
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;
  guint num_users = 0;

  GKeyFile *cf = g_key_file_new ();

  if (!g_key_file_load_from_file (cf, config_file, G_KEY_FILE_NONE, &error))
  {
    g_print ("Failed to load config file file: %s %s",config_file, error->message);
    return 0;
  }

  groups = g_key_file_get_groups (cf, NULL);

  for (group = groups; *group; group++) {
    gboolean parse_err = FALSE;

   if (!strncmp (*group, CONFIG_GROUP_SOURCE, sizeof (CONFIG_GROUP_SOURCE) - 1)) {
      num_users++;
    }
  }

  if (cf) {
    g_key_file_free (cf);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }

  return num_users;
}

gboolean parse_src_config (StreamCtx *sctx, GKeyFile *key_file, gchar *config_file, gchar *group)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);

  for (key = keys; *key; key++)
  {
    if (!g_strcmp0 (*key, "uri"))
    {
      gchar *filename = (gchar *)g_key_file_get_string(key_file, group, "uri", &error);
      if (g_str_has_prefix(filename, "file:///"))
      {
        sctx->uri = filename;
      }
      else
      {
        char *path = realpath(filename + 7, NULL);

        if (path == NULL)
        {
          printf("cannot find file with name[%s]\n", filename);
        }
        else
        {
          printf("Input file [%s]\n", path);
          sctx->uri = g_strdup_printf("file://%s", path);
          free(path);
        }
      }
      CHECK_PARSE_ERROR(error);
    }
  }
  return 0;
}

gboolean parse_sink_config(StreamCtx *sctx, GKeyFile *key_file, gchar *config_file, gchar *group)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);


  for (key = keys; *key; key++)
  {
    if (!g_strcmp0 (*key, "enable_playback"))
    {
      sctx->audio_config.enable_playback  = g_key_file_get_integer (key_file, group, "enable_playback", &error);
      CHECK_PARSE_ERROR (error);
    }

    if (!g_strcmp0 (*key, "asr_output_file_name"))
    {
      sctx->audio_config.asr_output_file_name  = (gchar *)g_key_file_get_string (key_file, group, "asr_output_file_name", &error);
      CHECK_PARSE_ERROR (error);
    }

    if (!g_strcmp0 (*key, "sync"))
    {
      sctx->audio_config.sync  = g_key_file_get_integer (key_file, group, "sync", &error);
      CHECK_PARSE_ERROR (error);
    }
  }
  return 0;
}

gboolean parse_config_file (AppCtx *appctx, gchar *config_file)
{
  GError *error = NULL;
  gboolean ret = 0;
  gchar **groups = NULL;
  gchar **group;
  guint num_users = 0;
  int i = 0;
  StreamCtx *sctx = NULL;

  GKeyFile *cf = g_key_file_new ();

  if (!g_key_file_load_from_file (cf, config_file, G_KEY_FILE_NONE, &error))
  {
    g_print ("Failed to load config file file: %s %s",config_file, error->message);
    return 1;
  }

  groups = g_key_file_get_groups (cf, NULL);

  for (group = groups; *group; group++) {
    gboolean parse_err = FALSE;

    /* parse source group */
    if (!strncmp (*group, CONFIG_GROUP_SOURCE, sizeof (CONFIG_GROUP_SOURCE) - 1))
    {
      sctx = &appctx->sctx[i];
      ret = parse_src_config (sctx, cf, config_file, *group);
      if(ret)
      {
        goto done;
      }
    }

    /* parse sink group */
    /* Increment the stream counter when both source and sink group are present */
    if (!strncmp (*group, CONFIG_GROUP_SINK, sizeof (CONFIG_GROUP_SINK) - 1))
    {
      sctx = &appctx->sctx[i];
      ret = parse_sink_config (sctx, cf, config_file, *group);
      if(ret)
      {
        goto done;
      }
      i++;
    }

  }
done:

  if (cf) {
    g_key_file_free (cf);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }
  if(ret)
  {
    return 1;
  }
  return 0;
}
