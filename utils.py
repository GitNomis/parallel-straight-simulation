import fileinput
import os
import sys
import warnings
from functools import reduce
from itertools import product, takewhile

import matplotlib.pyplot as plt
import numpy as np
from pgmpy.extern import tabulate
from pgmpy.inference import ApproxInference as RejSamp
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.models import BayesianNetwork
from tqdm import trange

from StraightSimulation import StraightSimulation as StrSim


def displayNetwork(network):
    network.to_daft(node_pos='spring').render()
    plt.show()


def _try_get_value(factor, state):
    for var, val in state:
        if val not in factor.state_names[var]:
            warnings.warn(f"{val} of {var} not in factor.")
            return 0
    return factor.get_value(**dict(state))


def getValues(factor, variables, state_names):
    vars = [[(var, val) for val in state_names[var]]
            for var in variables]
    arr = [_try_get_value(factor, state) for state in product(*vars)]
    card = tuple(len(state_names[var]) for var in variables)
    return np.array(arr).reshape(card)


def kullbackLeibler(p, q):
    if np.any((q == 0) & (p != 0)):
        warnings.warn(
            "Q(x) = 0 -> P(x) = 0 does not hold (approx = 0 -> exact = 0). Relative entropy not defined.")
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


def runJob(file):
    valid_methods = {"StrSim": StrSim, "RejSamp": RejSamp}
    valid_types = {'i': int, 'b': bool, 's': str}
    lines = fileinput.input(f"input/{file}.in")
    jobs = [(i, takewhile(lambda x: not x.isspace(), lines))
            for i in range(int(next(lines)))]

    for i, job in jobs:
        func, *func_args = next(job).split()
        if func == "skip":
            l = 0
            for _ in job:
                l += 1
            print(f"Skipped j{i} of {l} lines")
        elif func == "varianceTest":
            network = BayesianNetwork.load(f"networks/{func_args[0]}.bif")
            n_runs = int(func_args[1])
            ident = func_args[2] if len(func_args) > 2 else f"j{i}_"
            variables, evidence = next(job).split("|")
            args = {'variables': [], 'evidence': {}}
            for var in variables.split():
                args['variables'].append(var)
            for evi in evidence.split():
                var, val = evi.split("=")
                args['evidence'][var] = val
            methods, pars = [], []
            for line in job:
                method, *raw_pars = line.split()
                methods.append(valid_methods[method](network))
                par = {}
                for raw_par in raw_pars:
                    key, val = raw_par.split("=")
                    par[key] = valid_types[val[0]](val[1:])
                pars.append(par)
            varianceTest(network, methods, pars, args, n_runs,
                         f"output/jobs/{file}/{func_args[0]}", ident)
    lines.close()


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


def varianceTest(network, methods, pars, args, n_runs, outPath=None, ident=""):
    metrics = [kullbackLeibler, absError, relError]
    labels = ["relative entropy", "absolute error", "relative error"]

    titles = []
    for method, par in zip(methods, pars):
        if isinstance(method, StrSim):
            title = "Straight Simulation"
        elif isinstance(method, RejSamp):
            title = "Rejection Sampling"
            par['show_progress'] = False
        else:
            title = "Unknown"
        if 'parallel' in par and par['parallel']:
            title = "Parallel "+title
        titles.append(title)

    suptitle_method = f" using {titles[0]}" if np.unique(
        titles).size is 1 else ""
    file_method = f"_{titles[0].replace(' ','')}" if np.unique(
        titles).size is 1 else ""

    ns_samples = [par['n_samples'] for par in pars]

    suptitle_samples = f"{ns_samples[0]} samples, " if np.unique(
        ns_samples).size is 1 else ""
    file_samples = f"_{ns_samples[0]}s" if np.unique(
        ns_samples).size is 1 else ""

    if not suptitle_samples:
        titles = [f"{s} samples"if suptitle_method else f"{t}\nwith {s} samples" for t, s in zip(
            titles, ns_samples)]

    suptitle_vars = reduce(lambda x, y: x+', '+y, args['variables'])
    suptitle_evis = f"| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}" if args['evidence'] else ""
    plot_suptitle = f"P({suptitle_vars}{suptitle_evis}){suptitle_method} with {suptitle_samples}{n_runs} runs"

    ve = VarElim(network).query(**args)
    exact = getValues(ve, args['variables'], ve.state_names)

    metric_data = np.zeros((n_runs, len(methods), len(metrics)))
    value_data = np.zeros((n_runs, len(methods), exact.size))
    for i in trange(n_runs, desc="Running iteration", leave=None):
        for q, (method, par) in enumerate(zip(methods, pars)):
            query = method.query(**args, **par)
            values = getValues(query, args['variables'], ve.state_names)
            value_data[i, q] = values.flatten()
            for m, metric in enumerate(metrics):
                metric_data[i, q, m] = metric(exact, values)

    fig, axs = plt.subplots(nrows=len(metrics), ncols=1, sharex=True,
                            constrained_layout=True, figsize=(max(5, len(methods)*2), len(metrics)*3))
    for m in range(len(metrics)):
        axs[m].violinplot(metric_data[:, :, m],
                          showmeans=True, showmedians=False)
        axs[m].tick_params(axis='x', direction='in',
                           bottom=True, top=True, labelbottom=False)
        axs[m].tick_params(axis='y', labelsize='small')
        axs[m].set_ylabel(labels[m], fontsize='small')
        axs[m].grid(axis='y', alpha=0.3, linestyle=':')
    axs[0].tick_params(labeltop=True, labelsize='small')
    axs[0].set_xticks(ticks=range(1, 1+len(methods)), labels=titles)

    fig.suptitle(plot_suptitle, fontsize='medium')
    if outPath:
        os.makedirs(outPath, exist_ok=True)
        plt.savefig(
            f"{outPath}/{ident}{outPath.split('/')[-1]}{file_method}{file_samples}_{n_runs}r_metrics.pdf", format='pdf', transparent=True)
    else:
        plt.show(block=False)

    states = [reduce(lambda x, y: x+',\n'+y, state) for state in product(*[[f"{var}={val}" for val in ve.state_names[var]]
                                                                           for var in args['variables']])]

    fig, axs = plt.subplots(nrows=len(states), ncols=1, sharex=True,
                            constrained_layout=True, figsize=(max(5, len(methods)*2), len(states)*3))
    for m in range(len(states)):
        axs[m].violinplot(value_data[:, :, m],
                          showmeans=True, showmedians=False)
        axs[m].tick_params(axis='x', direction='in',
                           bottom=True, top=True, labelbottom=False)
        axs[m].tick_params(axis='y', labelsize='small')
        axs[m].set_ylabel(states[m], fontsize='x-small')
        axs[m].grid(axis='y', alpha=0.3, linestyle=':')
        axs[m].hlines(exact.flatten()[m], [0.75], [len(methods)+0.25], linewidth=1,
                      color='r', alpha=0.5, label=f"exact = {exact.flatten()[m]:.4f}")
        axs[m].legend(fontsize='x-small')
    axs[0].tick_params(labeltop=True, labelsize='small')
    axs[0].set_xticks(ticks=range(1, 1+len(methods)), labels=titles)
    fig.suptitle(plot_suptitle, fontsize='medium')
    if outPath:
        plt.savefig(
            f"{outPath}/{ident}{outPath.split('/')[-1]}{file_method}{file_samples}_{n_runs}r_probs.pdf", format='pdf', transparent=True)
    else:
        plt.show()


def dump(methods, results, args=None, outPath=None):
    if outPath:
        stdoutSave = sys.stdout
        sys.stdout = open(
            f"{outPath}{'_'+str(args['n_samples'])+'s' if args else ''}.out", 'w')
    if args:
        print(
            f"Query: P({reduce(lambda x,y: x+', '+y, args['variables'])}| {reduce(lambda x,y: x+', '+y, [k+'='+v for k,v in args['evidence'].items()])}) with {args['n_samples']} samples")
    ve = results[0]
    exact = getValues(ve, args['variables'], ve.state_names)
    values = [getValues(res, args['variables'], ve.state_names)
              for res in results]
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
