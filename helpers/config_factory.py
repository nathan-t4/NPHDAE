from configs.lc1 import get_lc1_config
from configs.rlc import get_rlc_config
from configs.dgu import get_dgu_config
from configs.coupled_lc import get_coupled_lc_config
from configs.alternator import get_alternator_config
from configs.comp_circuits_old import get_comp_gnn_old_config
from configs.comp_circuits import get_comp_gnn_config
from configs.reuse_model import get_reuse_model_config

configs = {
    'LC1': get_lc1_config,
    'RLC': get_rlc_config,
    'DGU': get_dgu_config,
    'CoupledLC': get_coupled_lc_config,
    'Alternator': get_alternator_config,
    'CompCircuitsOld': get_comp_gnn_old_config,
    'CompCircuits': get_comp_gnn_config,
    'ReuseModel': get_reuse_model_config,
}

def config_factory(name, args):
    return configs[name](args)