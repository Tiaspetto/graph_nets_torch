from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from pickle import GLOBAL

from sklearn.feature_selection import SelectFdr

NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = 'senders'
GLOBALS = 'globals'
N_NODE = "n_node"
N_EDGE = "n_edge"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS)
GRAPH_NUMBER_FILEDS = (N_NODE, N_EDGE)
ALL_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE)

class GraphsTuple(
    collections.namedtuple("GraphsTuple", GRAPH_DATA_FIELDS + GRAPH_NUMBER_FILEDS)):
    
    def _validate_none_fields(self):
        if self.n_node is None:
            raise ValueError("Field 'n_node' cannot be None")
        if self.n_edge is None:
            raise ValueError("Field 'n_edge' cannot be None")
        if self.receovers is None and self.senders is not None:
            raise ValueError("Field 'senders' must be None as field 'receivers' is None")
        if self.senders is None and self.receivers is not None:
            raise ValueError("Field 'receivers' must be None as field 'senders' is None")
        if self.receivers is None and self.edges is not None:
            raise ValueError("Field 'edges' must be None as field 'receivers' and 'senders' are None ")
        
    def _init_(self, *args, **kwargs):
        del args, kwargs
        super(GraphsTuple, self).__init__()
        self._validate_none_fields()
        
    def replace(self, **kwargs):
        output = self._replace(**kwargs)
        output._validate_none_fields()
        return output
    
    def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
        return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})
            
