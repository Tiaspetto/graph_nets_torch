from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email import message
from operator import index

from numpy import indices

from graph_nets_torch import graphs
from graph_nets_torch import utils_torch

import torch
import torch.nn as nn

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBAL = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

def _validate_graph(graph, mandatory_field, additional_message=None):
    for field in mandatory_field:
        if getattr(graph, field) is None:
            message = "`{}` field cannot be None".format(field)
            if additional_message:
                message += " " + format(additional_message)
            message += "."
            raise ValueError(message)
        
def _validate_broadcasted_graph(graph, from_field, to_field):
    addtional_message = "when broadcasting {} to {}".format(from_field, to_field)
    _validate_graph(graph, [from_field, to_field], addtional_message)

def _get_static_num_nodes(graph):
    return None if graph.nodes is None else graph.nodes.size().tolist()[0]

def _get_static_num_edges(graph):
    return None if graph.senders is None else graph.senders.size().tolist()[0]

def broadcast_globals_to_edges(graph, name="broadcast_globals_to_edges", num_edges_hint=None):
    _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
    return utils_torch.repeat(graph.globíals, graph.n_edge, axis=0,
                           sum_repeats_hint=num_edges_hint)
    
def broadcast_globals_to_nodes(graph, name="broadcast_globals_to_nodes", num_nodes_hint=None):
    _validate_broadcasted_graph(graph, GLOBALS, N_NODE)
    return utils_torch.repeat(graph.globals, graph.n_node, axis=0, sum_repeats_hint=num_nodes_hint)

def broadcast_sender_nodes_to_edges(graph, name="broadcast_sender_nodes_to_edges"):
    _validate_broadcasted_graph(graph, NODES, SENDERS)
    return torch.gather(graph.nodes, index=graph.senders)

def broadcast_receiver_nodes_to_edges(graph, name="broadcast_receiver_nodes_to_edges"):
    _validate_broadcasted_graph(graph, NODES, RECEIVERS)
    return torch.gather(graph.nodes, index=graph.receivers)

class EdgesToGlobalsAggregator(nn.Module):
    def __init__(self, reducer, name="edges_to_globals_aggregator"):
        super(EdgesToGlobalsAggregator, self).__init__(name=name)
        self._reducer = reducer
    
    def forward(self, graph):
        _validate_graph(graph, (EDGES,), additional_message="when aggregating from edges.")
        num_graphs = utils_torch.get_num_graphs(graph)
        graph_index = torch.range(0, num_graphs)
        indices = utils_torch.repeat(graph_index, graph.n_node, axis=0, sum_repeats_hint=_get_static_num_nodes(graph))
        
        return self._reducer(graph.edges, indices, num_graphs)

class NodesToGlobalsAggregator(nn.Module):

    def __init__(self, reducer, name="nodes_to_globals_aggregator"):
        super(NodesToGlobalsAggregator, self).__init__(name=name)
        self._reducer = reducer

    def forward(self, graph):
        _validate_graph(graph, (NODES,), additional_message="when aggregating from nodes.")
        num_graphs = utils_torch.get_num_graphs(graph)
        graph_index = torch.range(num_graphs)
        indices = utils_torch.repeat(graph_index, graph.n_node, axis=0, sum_repeats_hint=_get_static_num_nodes(graph))
        return self._reducer(graph.nodes, indices, num_graphs)


class _EdgesToNodesAggregator(nn.Module):
    def __init__(self, reducer, use_sent_edges=False, name="edges_to_nodes_aggregator"):
        super(_EdgesToNodesAggregator, self).__init__(name=name)
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges

    def forward(self, graph):
        _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                        additional_message="when aggregating from edges.")
        if graph.nodes is not None and graph.nodes.size()[0] is not None:
            num_nodes = graph.nodes.size()[0]
        else:
            num_nodes = torch.sum(graph.n_node)
            indices = graph.senders if self._use_sent_edges else graph.receivers
            return self._reducer(graph.edges, indices, num_nodes)
        
        
class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):

    def __init__(self, reducer, name="sent_edges_to_nodes_aggregator"):
        super(SentEdgesToNodesAggregator, self).__init__(
            use_sent_edges=True,
            reducer=reducer,
            name=name)

class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):

    def __init__(self, reducer, name="received_edges_to_nodes_aggregator"):
        super(ReceivedEdgesToNodesAggregator, self).__init__(
            use_sent_edges=False, reducer=reducer, name=name)
        
def _unsorted_segment_reduction_or_zero(reducer, values, indices, num_groups):
    reduced = reducer(values, indices, num_groups)
    present_indices =utils_torch.unsorted_segment_max(torch.ones_like(indices, dtype=reduced.dtype), indices, num_groups)
    present_indices = torch.clip(present_indices, 0, 1)
    present_indices = torch.reshape(present_indices, [num_groups] + [1] * (len(reduced.size()) - 1))
    reduced *= present_indices
    return reduced

def unsorted_segment_min_or_zero(values, indices, num_groups, name="unsorted_segment_min_or_zero"):
    return _unsorted_segment_reduction_or_zero(utils_torch.unsorted_segment_min, values, indices, num_groups)

def unsorted_segment_max_or_zero(values, indices, num_groups, name="unsorted_segment_max_or_zero"):
    return _unsorted_segment_reduction_or_zero(utils_torch.unsorted_segment_max, values, indices, num_groups)

class EdgeBlock(nn.Module):
    def __init__(self,
                edge_model_fn,
                use_edges=True,
                use_receiver_nodes=True,
                use_sender_nodes=True,
                use_globals=True,
                name="edge_block"):
        
        super(EdgeBlock, self).__init__(name=name)

        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                                "use_receiver_nodes or use_globals must be True.")

        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

        self._edge_model = edge_model_fn()

    def forward(self, graph, edge_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}

        _validate_graph(graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")

        edges_to_collect = []

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges)

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

        if self._use_globals:
            num_edges_hint = _get_static_num_edges(graph)
            edges_to_collect.append(
                broadcast_globals_to_edges(graph, num_edges_hint=num_edges_hint))

        collected_edges = torch.concat(edges_to_collect, axis=-1)
        updated_edges = self._edge_model(collected_edges, **edge_model_kwargs)
        return graph.replace(edges=updated_edges)
    

class NodeBlock(nn.Module):

    def __init__(self,
                node_model_fn,
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=True,
                received_edges_reducer=utils_torch.unsorted_segment_max,
                sent_edges_reducer=utils_torch.unsorted_segment_max,
                name="node_block"):

        super(NodeBlock, self).__init__(name=name)

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError("At least one of use_received_edges, use_sent_edges, "
                            "use_nodes or use_globals must be True.")

        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._node_model = node_model_fn()
            if self._use_received_edges:
                if received_edges_reducer is None:
                    raise ValueError(
                        "If `use_received_edges==True`, `received_edges_reducer` "
                        "should not be None.")
                self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(received_edges_reducer)
            if self._use_sent_edges:
                if sent_edges_reducer is None:
                    raise ValueError(
                        "If `use_sent_edges==True`, `sent_edges_reducer` "
                        "should not be None.")
                self._sent_edges_aggregator = SentEdgesToNodesAggregator(sent_edges_reducer)

    def forward(self, graph, node_model_kwargs=None):
        if node_model_kwargs is None:
            node_model_kwargs = {}

        nodes_to_collect = []

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            # The hint will be an integer if the graph has node features and the total
            # number of nodes is known at tensorflow graph definition time, or None
            # otherwise.
            num_nodes_hint = _get_static_num_nodes(graph)
            nodes_to_collect.append(
                broadcast_globals_to_nodes(graph, num_nodes_hint=num_nodes_hint))

        collected_nodes = torch.concat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes, **node_model_kwargs)
        return graph.replace(nodes=updated_nodes)
    
class GlobalBlock(nn.Module):
    def __init__(self,
                global_model_fn,
                use_edges=True,
                use_nodes=True,
                use_globals=True,
                nodes_reducer=utils_torch.unsorted_segment_sum,
                edges_reducer=utils_torch.unsorted_segment_sum,
                name="global_block"):
        super(GlobalBlock, self).__init__(name=name)

        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                            "use_nodes or use_globals must be True.")

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                if edges_reducer is None:
                    raise ValueError(
                        "If `use_edges==True`, `edges_reducer` should not be None.")
                self._edges_aggregator = EdgesToGlobalsAggregator(
                    edges_reducer)
            if self._use_nodes:
                if nodes_reducer is None:
                    raise ValueError(
                            "If `use_nodes==True`, `nodes_reducer` should not be None.")
                self._nodes_aggregator = NodesToGlobalsAggregator(
                    nodes_reducer)

    def forward(self, graph, global_model_kwargs=None):
        if global_model_kwargs is None:
            global_model_kwargs = {}

        globals_to_collect = []

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            globals_to_collect.append(self._edges_aggregator(graph))

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            globals_to_collect.append(self._nodes_aggregator(graph))

        if self._use_globals:
            _validate_graph(graph, (GLOBALS,), "when use_globals == True")
            globals_to_collect.append(graph.globals)

        collected_globals = torch.concat(globals_to_collect, axis=-1)
        updated_globals = self._global_model(
            collected_globals, **global_model_kwargs)
        return graph.replace(globals=updated_globals)