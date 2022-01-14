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

CUSTOM_LIB:=libnvds_custom_sequence_preprocess.so

CC = g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CFLAGS:= -fvisibility=hidden -Wall -Werror

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

PKGS:= gstreamer-1.0

CFLAGS+= -I../../../../includes \
         -I ../../../../gst-plugins/gst-nvdspreprocess/include \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)


CUSTOM_LIB_SRCS:= $(wildcard ./*.cpp ./*.cu)
CUSTOM_LIB_INCS:= $(wildcard ./*.h)
CUSTOM_LIB_OBJS:= $(CUSTOM_LIB_SRCS:.cpp=.o)
CUSTOM_LIB_OBJS:= $(CUSTOM_LIB_OBJS:.cu=.o)

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group


all: $(CUSTOM_LIB)

$(CUSTOM_LIB): $(CUSTOM_LIB_OBJS) Makefile
	$(CC) -o $(CUSTOM_LIB) $(CUSTOM_LIB_OBJS) $(LFLAGS)


%.o: %.cpp $(CUSTOM_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<
%.o: %.cu $(CUSTOM_LIB_INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

install: $(CUSTOM_LIB)
	cp -rv $(CUSTOM_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(CUSTOM_LIB_OBJS) $(CUSTOM_LIB)

