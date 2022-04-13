from itertools import product
from functools import reduce
from pgmpy.extern import tabulate
from tqdm import trange
from pgmpy.factors.discrete import DiscreteFactor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import daft
import sys
import warnings

class Logger(object):
    def __init__(self):
        pass

def displayNetwork(network):
    network.to_daft(node_pos='spring').render()
    plt.show()
    

def getValues(factor,variables,state_names):
    vars = [[(var, val) for val in state_names[var]]
            for var in variables]         
    arr=[factor.get_value(**dict(state)) for state in product(*vars)]    
    card = tuple(len(state_names[var]) for var in variables)    
    return np.array(arr).reshape(card)            

def kullbackLeibler(p,q):
    if np.any((q==0)&(p!=0)):
        warnings.warn("Q(x) = 0 -> P(x) = 0 does not hold. Relative entropy not defined.")
        return np.nan
    return np.sum(p*np.nan_to_num(np.log(p/q)))
    
def _absError(exact, approx, state):
    state = dict(state)
    return abs(exact.get_value(**state)-approx.get_value(**state))


def absError(exact, approx):
    error = 0
    vars = [[(var, val) for val in exact.state_names[var]]
            for var in exact.scope()]
    for state in product(*vars):
        error += _absError(exact, approx, state)
    return error


def _relError(exact, approx, state):
    state = dict(state)
    return abs(1-(approx.get_value(**state)/exact.get_value(**state)))


def relError(exact, approx):
    error = 0
    vars = [[(var, val) for val in exact.state_names[var]]
            for var in exact.scope()]
    for state in product(*vars):
        error += _relError(exact, approx, state)
    return error


def multiRun(exact, method, args, n_runs):
    vars = [[(var, val) for val in exact.state_names[var]]
            for var in exact.scope()]
    data = np.zeros((exact.values.size, 2, n_runs))
    for i in trange(n_runs):
        cpd = method.query(**args, show_progress=False)
        for j, state in enumerate(product(*vars)):
            data[j][0][i] = _absError(exact, cpd, state)
            data[j][1][i] = _relError(exact, cpd, state)
    means = data.mean(2)
    stds = data.std(2)
    states = list(
        zip(*product(*[[val for _, val in states] for states in vars])))
    table = np.stack([*states, means[:, 0], stds[:, 0],
                     means[:, 1], stds[:, 1]], 1)
    print(
        tabulate(table, [*cpd.scope(), "abs error mean", "abs error std", "rel error mean", "rel error std"], "grid"))

def table(ve,sts,stsp,rs):
    pass

def dump(ve, sts, stsp, rs, args=None, n_samples=None, outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(outPath, 'w')
    if args and n_samples:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {n_samples} samples")

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

def test(ve,sts,stsp,rs,args=None,n_samples=None,outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(outPath, 'w')
    if args and n_samples:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {n_samples} samples")

    veVal = ve.values
    stsVal = getValues(sts,ve.variables,ve.state_names)
    stspVal = getValues(stsp,ve.variables,ve.state_names)
    rsVal = getValues(rs,ve.variables,ve.state_names)
    data = [["Straight Simulation",kullbackLeibler(veVal,stsVal) ,np.sum(np.abs(veVal-stsVal)), np.sum(np.abs(1-(stsVal/veVal)))],
            ["Parallel Straight Simulation",kullbackLeibler(veVal,stspVal), np.sum(np.abs(veVal-stspVal)), np.sum(np.abs(1-(stspVal/veVal)))],
            ["Rejection Sampling", kullbackLeibler(veVal,rsVal),np.sum(np.abs(veVal-rsVal)),np.sum(np.abs(1-(rsVal/veVal)))]]

    print(
        tabulate(data, ["method","relative entropy", "absolute error", "relative error"], "grid"))
    states = list(
        zip(*product(*[ ve.state_names[var] for var in ve.variables])))
    data = np.stack([*states,veVal.ravel(),stsVal.ravel(),stspVal.ravel(),rsVal.ravel()],1) 
    print(tabulate(data,[*ve.variables,"Variable Elimination","Straight Simulation","Parallel Straight Simulation","Rejection Sampling"],"grid"))       
    print(f"Variable Elimination:\n{ve}")
    print(f"Straight Simulation:\n{sts}")
    print(f"Parallel Straight Simulation:\n{stsp}")
    print(f"Rejection Sampling:\n{rs}")
    if outPath:
        sys.stdout.close()
        sys.stdout = stdoutSave        
