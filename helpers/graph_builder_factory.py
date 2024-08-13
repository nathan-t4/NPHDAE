from scripts.graph_builders import *

gb = {
    'LC1': LC1GraphBuilder,
    'LC2': LC2GraphBuilder,
    'RLC': TestGraphBuilder,
    'DGU': TestGraphBuilder, # DGUGraphBuilder
    'CoupledLC': CoupledLCGraphBuilder,
    'Alternator': AlternatorGraphBuilder,
}

def gb_factory(name):
    return gb[name]