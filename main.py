from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.inference import ApproxInference as RejSamp
from StraightSimulation import StraightSimulation as StrSim
from StraightSimulation import Logger as StrSimLog
from utils import *

file = 'diamonds/basic10_0.5'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}' if 1 else None


def main():
    methods = ["Variable Elimination","Straight Simulation","Parallel Straight Simulation","Rejection Sampling"]
    args = {'variables':['V0'],'evidence':{'V9':'DiTernary'}}
    #args = {'variables':[ 'MaryCalls','JohnCalls'], 'evidence': {'Earthquake':'True','Burglary':'True'}}
    #args = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
    #args = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
    n_samples = 20
    n_runs = 100
    network = BN.load(inPath)
        
    #displayNetwork(network)
    test(methods, args, n_samples, network)
    del args['n_samples']
    varianceTest(network,args,n_samples,n_runs,outPath)
    # strSim = StrSimLog(network)
    # print(strSim.query(**args,n_samples=n_samples))
    # strSim.plot()
    # print(*strSim.getLogs())
    

def test(methods, args, n_samples, network):
    varElim = VarElim(network)
    strSim = StrSim(network)
    rejSamp = RejSamp(network)
    results = []
    results.append(varElim.query(**args))
    args['n_samples']=n_samples
    results.append(strSim.query(**args))
    results.append(strSim.query(**args,parallel=True))
    try:
        results.append(rejSamp.query(**args))
    except:
        del methods[-1]
    dump(methods,results,args,outPath)


if __name__ == "__main__":
    main()
