from itertools import product
from functools import reduce
from pgmpy.extern import tabulate
import matplotlib.pyplot as plt
import daft
import sys


def displayNetwork(network):
    network.to_daft(node_pos='spring').render()
    plt.show()


def absError(exact, approx):
    error = 0
    vars = [[(var, val) for val in exact.state_names[var]]
            for var in exact.scope()]
    for state in product(*vars):
        state = dict(state)
        error += abs(exact.get_value(**state)-approx.get_value(**state))
    return error


def relError(exact, approx):
    error = 0
    vars = [[(var, val) for val in exact.state_names[var]]
            for var in exact.scope()]
    for state in product(*vars):
        state = dict(state)
        error += abs(1-(approx.get_value(**state)/exact.get_value(**state)))
    return error


def dump(ve, sts, stsp, rs, par=None, n_samples=None, outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(outPath, 'w')
    if par and n_samples:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, par['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in par['evidence'].items()])}) with {n_samples} samples")

    data = [["Straight Simulation", absError(ve, sts), relError(ve, sts)],
            ["Parallel Straight Simulation", absError(
                ve, stsp), relError(ve, stsp)],
            ["Rejection Sampling", absError(ve, rs), relError(ve, rs)]]

    print(
        tabulate(data, ["method", "absolute error", "relative error"], "grid"))
    print(f"Variable Elimination:\n{ve}")
    print(f"Straight Simulation:\n{sts}")
    print(f"Parallel Straight Simulation:\n{stsp}")
    print(f"Rejection Sampling:\n{rs}")
    if outPath:
        sys.stdout.close()
        sys.stdout = stdoutSave
