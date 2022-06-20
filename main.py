from pgmpy.inference import ApproxInference as RejSamp
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.models import BayesianNetwork as BN

from StraightSimulation import StraightSimulation as StrSim
from utils import *

file = 'earthquake'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}' if 0 else None


def main():
    network = BN.load(inPath)
    n_samples = 10000
    n_runs = 100
    methods = [StrSim(network), StrSim(network), RejSamp(network)]
    pars = [{'n_samples': n_samples},
            {'parallel': True, 'n_samples': n_samples},
            {'n_samples': n_samples}]
    args = {'variables': ['Alarm'], 'evidence': {}}
    varianceTest(network, methods, pars, args, n_runs, outPath)
    #runJob(input())

def test(args, n_samples, network):
    methods = ["Variable Elimination", "Straight Simulation",
               "Parallel Straight Simulation", "Rejection Sampling"]
    varElim = VarElim(network)
    strSim = StrSim(network)
    rejSamp = RejSamp(network)
    results = []
    results.append(varElim.query(**args))
    args['n_samples'] = n_samples
    results.append(strSim.query(**args))
    results.append(strSim.query(**args, parallel=True))
    try:
        results.append(rejSamp.query(**args))
    except:
        del methods[-1]
    dump(methods, results, args, outPath)


if __name__ == "__main__":
    main()
