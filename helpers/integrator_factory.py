import sys
sys.path.append('..')
from integrators.rk4 import rk4
from integrators.euler_variants import euler
from integrators.adam_bashforth import adam_bashforth

integrators = {
    'rk4': rk4,
    'euler': euler,
    'adam_bashforth': adam_bashforth,
}

def integrator_factory(name):
    return integrators[name]