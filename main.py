from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.inference import ApproxInference as RejSamp
from StraightSimulation import StraightSimulation as StrSim
from StraightSimulation import Logger as StrSimLog
from utils import *

file = 'diamonds/basic10_0.5'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}' if 0 else None


def main():
    network = BN.load(inPath)
    titles = ["Straight Simulation","Parallel Straight Simulation","Rejection Sampling"]
    methods = [StrSim(network),StrSim(network),RejSamp(network)]
    pars = [{},{'parallel':True},{}]
    args = {'variables':['V0'],'evidence':{'V9':'DiTernary'}}

    #args = {'variables':[ 'MaryCalls','JohnCalls'], 'evidence': {'Earthquake':'True','Burglary':'True'}}
    #args = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
    #args = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
    n_samples = 2
    n_runs = 10000
    network = BN.load(inPath)
        
    #displayNetwork(network)
    test(dict(args), n_samples, network)
    varianceTest(network,titles,methods,pars,args,n_samples,n_runs,outPath)
    # strSim = StrSimLog(network)
    # print(strSim.query(**args,n_samples=n_samples))
    # strSim.plot()
    # print(*strSim.getLogs())
    

def test(args, n_samples, network):
    methods = ["Variable Elimination","Straight Simulation","Parallel Straight Simulation","Rejection Sampling"]
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
