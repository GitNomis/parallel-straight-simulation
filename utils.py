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

def _absError(exact,approx):
    return np.abs(exact-approx)

def absError(exact, approx):
    return np.sum(np.abs(exact-approx))

def _relError(exact,approx):
    if np.any(exact==0):
        warnings.warn("Exact = 0. Relative error not defined.")
        return np.nan
    return np.abs(1-(approx/exact))

def relError(exact, approx):
    if np.any(exact==0):
        warnings.warn("Exact = 0. Relative error not defined.")
        return np.nan
    return np.sum(np.abs(1-(approx/exact)))

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

def dump(methods,results,args=None,outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(outPath, 'w')
    if args:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {args['n_samples']} samples")
    ve = results[0]
    exact = ve.values
    values = [getValues(res,ve.variables,ve.state_names) for res in results]
    data = []
    for method,value in zip(methods[1:],values[1:]):
        data.append([method,kullbackLeibler(exact,value),absError(exact,value),relError(exact,value)])
    print(
        tabulate(data, ["method","relative entropy", "absolute error", "relative error"], "grid"))
        
    states = list(
        zip(*product(*[ ve.state_names[var] for var in ve.variables])))
    data = np.stack([*states,*[val.ravel() for val in values]],1) 
    print(tabulate(data,[*ve.variables,*methods],"grid"))  
    for method,result in zip(methods,results):
        print(f"{method}:\n{result}")     
    
    if outPath:
        sys.stdout.close()
        sys.stdout = stdoutSave        
