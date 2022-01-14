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

#pragma once

#include <initializer_list>
#include <iostream>
#include <vector>

constexpr unsigned int NB_COARSE_LABEL_AUDIO = 8;

constexpr auto LABELS_AUDIO = {
    "1_engine",
    "2_machinery-impact",
    "3_non-machinery-impact",
    "4_powered-saw",
    "5_alert-signal",
    "6_music",
    "7_human-voice",
    "8_dog",
    "1-1_small-sounding-engine",
    "1-2_medium-sounding-engine",
    "1-3_large-sounding-engine",
    "2-1_rock-drill",
    "2-2_jackhammer",
    "2-3_hoe-ram",
    "2-4_pile-driver",
    "3-1_non-machinery-impact",
    "4-1_chainsaw",
    "4-2_small-medium-rotating-saw",
    "4-3_large-rotating-saw",
    "5-1_car-horn",
    "5-2_car-alarm",
    "5-3_siren",
    "5-4_reverse-beeper",
    "6-1_stationary-music",
    "6-2_mobile-music",
    "6-3_ice-cream-truck",
    "7-1_person-or-small-group-talking",
    "7-2_person-or-small-group-shouting",
    "7-3_large-crowd",
    "7-4_amplified-speech",
    "8-1_dog-barking-whining"};

/* This is a sample audio neural network parsing function. */

extern "C"
{
    bool NvDsInferParseCustomAudio(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                   std::vector<NvDsInferAttribute> &attrList, std::string &attrString);
}

std::vector<unsigned int> index_giver_subcategory(const char *label);
