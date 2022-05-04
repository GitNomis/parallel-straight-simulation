from itertools import product
from functools import reduce
from pgmpy.extern import tabulate
from tqdm import trange
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.inference import ApproxInference as RejSamp
from StraightSimulation import StraightSimulation as StrSim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import daft
import sys
import os
import warnings


def displayNetwork(network):
    network.to_daft(node_pos='spring').render()
    plt.show()


def getValues(factor, variables, state_names):
    vars = [[(var, val) for val in state_names[var]]
            for var in variables]
    arr = [factor.get_value(**dict(state)) for state in product(*vars)]
    card = tuple(len(state_names[var]) for var in variables)
    return np.array(arr).reshape(card)


def kullbackLeibler(p, q):
    if np.any((q == 0) & (p != 0)):
        warnings.warn(
            "Q(x) = 0 -> P(x) = 0 does not hold. Relative entropy not defined.")
        return np.nan
    return np.sum(p*np.nan_to_num(np.log(p/q)))


def _absError(exact, approx):
    return np.abs(exact-approx)


def absError(exact, approx):
    return np.sum(np.abs(exact-approx))


def _relError(exact, approx):
    if np.any(exact == 0):
        warnings.warn("Exact = 0. Relative error not defined.")
        return np.nan
    return np.abs(1-(approx/exact))


def relError(exact, approx):
    if np.any(exact == 0):
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


def varianceTest(network, args, n_samples, n_runs,outPath=None):
    methods = ["Straight Simulation","Parallel Straight Simulation","Rejection Sampling"]
    metrics = [kullbackLeibler, absError, relError]
    m_names = ["relative entropy", "absolute error", "relative error"]
    varElim = VarElim(network)
    strSim = StrSim(network)
    rejSamp = RejSamp(network)
    ve = varElim.query(**args)
    exact = getValues(ve,args['variables'],ve.state_names)

    args['n_samples'] = n_samples
    args['show_progress'] = False
    data = np.zeros((len(metrics)+ve.values.size, n_runs, 3))
    for i in trange(n_runs):
        queries = [strSim.query(**args), strSim.query(**
                                                      args, parallel=True), rejSamp.query(**args)]
        for v, query in enumerate(queries):
            values = getValues(query, args['variables'], ve.state_names)
            for m, metric in enumerate(metrics):
                data[m, i, v] = metric(exact, values)
            data[len(metrics):, i, v] = values.flatten()

    fig, axs = plt.subplots(nrows=len(metrics), ncols=1,sharex=True,constrained_layout=True,figsize=(5,9))
    for m in range(len(metrics)):
        axs[m].violinplot(data[m], showmeans=True, showmedians=False) 
        axs[m].tick_params(axis='x',direction='in',bottom=True,top=True,labelbottom=False)
        axs[m].tick_params(axis='y',labelsize='small')
        axs[m].set_ylabel(m_names[m],fontsize='small')
        axs[m].grid(axis = 'y', alpha=0.3,linestyle=':')
    axs[0].tick_params( labeltop=True,labelsize='small')
    axs[0].set_xticks(ticks=range(1,4),labels=methods)   
   
    fig.suptitle(f"P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {n_samples} samples, {n_runs} runs",fontsize='medium')
    if outPath:
        os.makedirs(outPath,exist_ok=True)
        plt.savefig(f"{outPath}/{outPath.split('/')[-1]}_{n_samples}s_{n_runs}r_metrics.pdf",format='pdf',transparent=True)
    else:    
        plt.show(block=False)

    states = [reduce(lambda x,y: x+',\n'+y,state) for state in product(*[[f"{var}={val}" for val in ve.state_names[var]]
            for var in args['variables']])]
            
    fig, axs = plt.subplots(nrows=len(states), ncols=1,sharex=True,constrained_layout=True,figsize=(5,2+len(states)*3))
    for m in range(len(states)):
        axs[m].violinplot(data[m+len(metrics)], showmeans=True, showmedians=False)  
        axs[m].tick_params(axis='x',direction='in',bottom=True,top=True,labelbottom=False)
        axs[m].tick_params(axis='y',labelsize='small')
        axs[m].set_ylabel(states[m],fontsize='x-small')
        axs[m].grid(axis = 'y', alpha=0.3,linestyle=':')
        axs[m].hlines(exact.flatten()[m], [0.75],[3.25],linewidth=1,color='r',alpha=0.5,label=f"exact = {exact.flatten()[m]:.4f}")
        axs[m].legend(fontsize='x-small') #xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
    axs[0].tick_params( labeltop=True,labelsize='small')
    axs[0].set_xticks(ticks=range(1,4),labels=methods)     
    fig.suptitle(f"P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {n_samples} samples, {n_runs} runs",fontsize='medium') 
    if outPath:
        plt.savefig(f"{outPath}/{outPath.split('/')[-1]}_{n_samples}s_{n_runs}r_probs.pdf",format='pdf')
    else:    
        plt.show(block=False)


def dump(methods, results, args=None, outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(f"{outPath}{'_'+str(args['n_samples'])+'s' if args else ''}.out", 'w')
    if args:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {args['n_samples']} samples")
    ve = results[0]
    exact = getValues(ve,args['variables'],ve.state_names)
    values = [getValues(res, args['variables'], ve.state_names) for res in results]
    data = []
    for method, value in zip(methods[1:], values[1:]):
        data.append([method, kullbackLeibler(exact, value),
                    absError(exact, value), relError(exact, value)])
    print(
        tabulate(data, ["method", "relative entropy", "absolute error", "relative error"], "grid"))

    states = list(
        zip(*product(*[ve.state_names[var] for var in args['variables']])))
    data = np.stack([*states, *[val.ravel() for val in values]], 1)
    print(tabulate(data, [*args['variables'], *methods], "grid"))
    for method, result in zip(methods, results):
        print(f"{method}:\n{result}")

    if outPath:
        sys.stdout.close()
        sys.stdout = stdoutSave
