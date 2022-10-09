
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from absl import logging
from graph_nets import graphs
import six
from six.moves import range
import torch 

from torch_scatter import scatter

NODES = graphs.NODES
EDGES = graphs.EDGES
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS


def _get_shape(tensor):
    shape_list = list(tensor.size())
    if all(s is not None for s in shape_list):
        return shape_list
    shape_tensor = torch.Size(tensor)
    return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]

def repeat(tensor, repeats, axis=0, name="repeat", sum_repeats_hint=None):
    if sum_repeats_hint is not None:
        sum_repeats = sum_repeats_hint
    else:
        sum_repeats = torch.sum(repeats)
        
    cumsum_repeats = torch.cumsum(repeats, exclusive=False)
    block_split_indicators = torch.sum(
            torch.nn.functional.one_hot(cumsum_repeats, sum_repeats), axis=0)
    gather_indices = torch.cumsum(block_split_indicators, exclusive=False)
    
    repeated_tensor = torch.gather(tensor, dim=axis, index=gather_indices)
    
    shape =tensor.size().tolist()
    shape[axis] = sum_repeats_hint
    repeated_tensor.reshape(shape)
    
    return repeated_tensor

def get_num_graphs(input_graphs, name="get_num_graphs"):
    return _get_shape(input_graphs.n_node)[0]


def unsorted_segment_sum(values, indices, num_groups, name="unsorted_segment_sum"):
    return  scatter(values, indices, reduce="sum")

def unsorted_segment_mean(values, indices, num_groups, name="unsorted_segment_mean"):
    return  scatter(values, indices, reduce="mean")

def unsorted_segment_max(values, indices, num_groups, name="unsorted_segment_max"):
    return  scatter(values, indices, reduce="max")

def unsorted_segment_min(values, indices, num_groups, name="unsorted_segment_min"):
    return  scatter(values, indices, reduce="min")
