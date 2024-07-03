from scripts.graph_builder import *

gb = {
    'LC1': LC1GraphBuilder,
    'LC2': LC2GraphBuilder,
    'CoupledLC': CoupledLCGraphBuilder,
    'Alternator': AlternatorGraphBuilder,
}

def gb_factory(name):
    return gb[name]