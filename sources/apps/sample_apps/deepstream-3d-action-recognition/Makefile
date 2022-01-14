################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

NVDS_VERSION:=6.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

APP:= deepstream-3d-action-recognition

CC = g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CFLAGS:= -fvisibility=hidden -Wall -Werror

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

CFLAGS+= -I../../../includes \
         -I ../../../gst-plugins/gst-nvdspreprocess/include \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)
APP_OBJS:= $(APP_SRCS:.cpp=.o)

SUBFOLDER:=custom_sequence_preprocess

all: $(APP) $(SUBFOLDER)

%.o: %.cpp $(APP_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(APP_OBJS) Makefile
	$(CC) -o $(APP) $(APP_OBJS) $(LIBS)

$(SUBFOLDER):
	$(MAKE) -C $@ $(MAKECMDGOALS)

install: $(APP) $(SUBFOLDER)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean: $(SUBFOLDER)
	rm -rf $(APP_OBJS) $(APP)

.PHONY: all $(SUBFOLDER)
