################################################################################
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

#!/bin/bash

TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    if [ "${OS}" = "rhel" ]; then
        BASE_LIB_DIR="/usr/lib64/"
    elif [ "${OS}" = "ubuntu" ]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
fi

PREV_DS_VER=6.0
if [ -n $PREV_DS_VER ]; then
  if [ "${PREV_DS_VER}" = "" ]; then
    echo "PREV_DS_VER not set"
    exit 1
  fi
fi

if [ ! -d /opt/nvidia/deepstream/deepstream-${PREV_DS_VER} ]; then
    echo "This version of DeepStream is not present in the system."
    exit 1
fi

if [ -n $TARGET_DEVICE ]; then
  if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    update-alternatives --remove deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libv4l/plugins/libcuvidv4l2_plugin.so
    update-alternatives --remove deepstream-v4l2library /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4l2.so
    update-alternatives --remove deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4lconvert.so
    update-alternatives --remove deepstream-asr-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-asr-app
    update-alternatives --remove deepstream-asr-tts-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-asr-tts-app
    update-alternatives --remove deepstream-avsync-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-avsync-app
    rm -rf /opt/riva/
    rm -rf /opt/kenlm/
    rm -rf $BASE_LIB_DIR/libv4lconvert.so.0.0.99999
    rm -rf $BASE_LIB_DIR/libv4l2.so.0.0.99999
  fi
fi
update-alternatives --remove deepstream-plugins /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/gst-plugins
update-alternatives --remove deepstream-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-app
update-alternatives --remove deepstream-audio /opt/nvidia/deepstream/deepstream-6.0/bin/deepstream-audio
update-alternatives --remove deepstream-test1-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test1-app
update-alternatives --remove deepstream-test2-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test2-app
update-alternatives --remove deepstream-test3-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test3-app
update-alternatives --remove deepstream-test4-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test4-app
update-alternatives --remove deepstream-test5-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test5-app
update-alternatives --remove deepstream-testsr-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-testsr-app
update-alternatives --remove deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-transfer-learning-app
update-alternatives --remove deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-user-metadata-app
update-alternatives --remove deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-dewarper-app
update-alternatives --remove deepstream-nvof-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvof-app
update-alternatives --remove deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-decode-app
update-alternatives --remove deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-gst-metadata-app
update-alternatives --remove deepstream-opencv-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-opencv-test
update-alternatives --remove deepstream-perf-demo /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-perf-demo
update-alternatives --remove deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-preprocess-test
update-alternatives --remove deepstream-segmentation-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-segmentation-app
update-alternatives --remove deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-meta-test
update-alternatives --remove deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-appsrc-test
update-alternatives --remove deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvdsanalytics-test
update-alternatives --remove deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-action-recognition
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/samples/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/sources/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/LicenseAgreement.pdf
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/LICENSE.txt
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/README
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/README.rhel
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/install.sh
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/uninstall.sh
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/version
ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/
