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

constexpr const char* kActionRecognition = "action-recognition";
constexpr const char* kUriList = "uri-list";
constexpr const char* kPreprocessConfig = "preprocess-config";
constexpr const char* kInferenceConfig = "infer-config";
constexpr const char* kMuxerHeight = "muxer-height";
constexpr const char* kMuxerWidth = "muxer-width";
constexpr const char* kMuxerBatchTimeout = "muxer-batch-timeout"; // usec

constexpr const char* kTilerHeight = "tiler-height";
constexpr const char* kTilerWidth = "tiler-width";
constexpr const char* kDisplaySync = "display-sync";

constexpr const char* kDebug = "debug";
constexpr const char* kEnableFps = "enable-fps";

#define PARSE_FAILED(statement, fmt, ...)               \
    if (!(statement)) {                                 \
      LOG_ERROR(fmt, ##__VA_ARGS__);                    \
      return false;                                     \
    }

#define PARSE_WITH_ERROR(statement, fmt, ...)           \
    do {                                                \
      statement;                                        \
      SafePtr<GError> ptrErr__(error, g_error_free); \
      if (error) {                                     \
        LOG_ERROR(fmt ", error msg: %s", ##__VA_ARGS__, ptrErr__->message); \
        return false;                                   \
      }                                                 \
    } while(0)

bool parse_action_config(const char* path, NvDsARConfig& config) {
  SafePtr<GKeyFile> keyfile(g_key_file_new(), g_key_file_free);
  SafePtr<gchar*> safeKeys(nullptr, g_strfreev);
  GError *error = nullptr;

  PARSE_WITH_ERROR(
    g_key_file_load_from_file (keyfile.get(), path, G_KEY_FILE_NONE, &error),
    "load config: %s failed", path);
  PARSE_FAILED(
    g_key_file_has_group (keyfile.get(), kActionRecognition),
    "parse config: %s failed, group: %s is missing", path, kActionRecognition);

  PARSE_WITH_ERROR(
    safeKeys.reset(g_key_file_get_keys (keyfile.get(), kActionRecognition, NULL, &error)),
    "parse keys of group: %s failed in config: %s", kActionRecognition, path
  );

  PARSE_WITH_ERROR(
    config.debug = (DebugLevel)g_key_file_get_integer (keyfile.get(), kActionRecognition,
      kDebug, &error),
    "parse key: %s failed in config: %s", kDebug, path);

  SafePtr<gchar*> uri_list(nullptr, g_strfreev);
  gsize num_strings = 0;
  PARSE_WITH_ERROR(
    uri_list.reset(g_key_file_get_string_list (keyfile.get(), kActionRecognition,
      kUriList, &num_strings, &error)),
    "parse key: %s failed in config: %s", kUriList, path);
  for (gsize i = 0; i < num_strings; ++i) {
    config.uri_list.push_back(uri_list.get()[i]);
  }
  g_assert(config.uri_list.size() > 0);

  SafePtr<gchar> config_str(nullptr, g_free);

  PARSE_WITH_ERROR(
    config_str.reset(g_key_file_get_string (keyfile.get(), kActionRecognition,
      kPreprocessConfig, &error)),
    "parse key: %s failed in config: %s", kPreprocessConfig, path);
  config.preprocess_config = config_str.get();

  PARSE_WITH_ERROR(
    config_str.reset(g_key_file_get_string (keyfile.get(), kActionRecognition,
      kInferenceConfig, &error)),
    "parse key: %s failed in config: %s", kInferenceConfig, path);
  config.infer_config = config_str.get();

  PARSE_WITH_ERROR(
    config.muxer_batch_timeout = g_key_file_get_integer (keyfile.get(), kActionRecognition,
      kMuxerBatchTimeout, &error),
    "parse key: %s failed in config: %s", kMuxerBatchTimeout, path);

  PARSE_WITH_ERROR(
    config.muxer_height = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kMuxerHeight, &error),
    "parse key: %s failed in config: %s", kMuxerHeight, path);

  PARSE_WITH_ERROR(
    config.muxer_width = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kMuxerWidth, &error),
    "parse key: %s failed in config: %s", kMuxerWidth, path);
  PARSE_WITH_ERROR(
    config.tiler_height = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kTilerHeight, &error),
    "parse key: %s failed in config: %s", kTilerHeight, path);
  PARSE_WITH_ERROR(
    config.tiler_width = (uint32_t)g_key_file_get_uint64 (keyfile.get(), kActionRecognition,
      kTilerWidth, &error),
    "parse key: %s failed in config: %s", kTilerWidth, path);

  PARSE_WITH_ERROR(
    config.display_sync = (uint32_t)g_key_file_get_boolean (keyfile.get(), kActionRecognition,
      kDisplaySync, &error),
    "parse key: %s failed in config: %s", kDisplaySync, path);

    PARSE_WITH_ERROR(
    config.enableFps = (uint32_t)g_key_file_get_boolean (keyfile.get(), kActionRecognition,
      kEnableFps, &error),
    "parse key: %s failed in config: %s", kDisplaySync, path);

  return true;
}