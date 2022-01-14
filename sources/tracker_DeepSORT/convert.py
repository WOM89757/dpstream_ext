################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

# This script convert DeepSORT's official re-identification model from TensorFlow
# frozen graph into UFF. The node edition is specific to the official model, since
# it contains nodes not suppported by TensorRT.

import graphsurgeon as gs
import tensorflow as tf
import uff
import sys

if len(sys.argv) < 2:
    print('Usage: python ' + sys.argv[0] + ' /path/to/model.pb')
    exit(0)

# load graph
filename_pb = sys.argv[1]
dynamic_graph = gs.DynamicGraph(filename_pb)
nodes = list(dynamic_graph.as_graph_def().node)

print('Converting...')
# create input node
input_node = gs.create_node("images",
        op="Placeholder",
        dtype=tf.float32,
        shape=[None, 128, 64, 3]
        )

# remove nodes in DeepSORT's re-identification model not supported by TensorRT,
# and connect with input node
for node in nodes:
    if "map" in node.name or "images" == node.name or "Cast" == node.name:
        dynamic_graph.remove(node)
    elif "conv1_1/Conv2D" == node.name:
        node.input.insert(0, "images")

# add input node to graph
dynamic_graph.append(input_node)

# create uff file
trt_graph = uff.from_tensorflow(dynamic_graph.as_graph_def())
filename_uff = filename_pb[:filename_pb.rfind('.')]  + '.uff'
print('Writing to disk...')
with open(filename_uff, 'wb') as f:
    f.write(trt_graph)
print('Saved as ' + filename_uff)
