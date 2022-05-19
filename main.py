from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VarElim
from pgmpy.inference import ApproxInference as RejSamp
from NetworkGenerator import chromatic_number
from StraightSimulation import StraightSimulation as StrSim
from StraightSimulation import Logger as StrSimLog
from networkx.algorithms.coloring import greedy_color 
from networkx.algorithms.moral import moral_graph
from utils import *

file = 'diamonds/basic5_0.2'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}' if 1 else None


def main():
    network = BN.load(inPath)
    n_samples = 100
    n_runs = 100
    methods = [StrSim(network),StrSim(network),RejSamp(network)]
    pars = [{'n_samples':n_samples},{'parallel':True,'n_samples':n_samples},{'n_samples':n_samples}]
    args = {'variables':['V0'],'evidence':{'V9':'DiTernary'}}
    # print(greedy_color(moral_graph(network)))
    # print(StrSim(network)._order())
    # print(chromatic_number(moral_graph(network),3))
    #args = {'variables':[ 'MaryCalls','JohnCalls'], 'evidence': {'Earthquake':'True','Burglary':'True'}}
    #args = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
    #args = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
    #displayNetwork(network)
    #test(dict(args), n_samples, network)
    varianceTest(network,methods,pars,args,n_runs,outPath)
    #job=input()
    #runJob(job)
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
