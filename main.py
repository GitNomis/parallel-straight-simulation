from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VE
from pgmpy.inference import ApproxInference as RS
from StraightSimulation import StraightSimulation as StS
from utils import *


file = 'hepar2'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}.out'
#par = {'variables':[ 'JohnCalls'], 'evidence': {'Earthquake':'True','Burglary':'True'}}
#par = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
par = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
n_samples = 10000


def main():
    network = BN.load(inPath)
    #displayNetwork(network)
    ve = VE(network)
    sts = StS(network)
    rs = RS(network)
    veRes=ve.query(**par)
    stsRes=sts.query(**par,n_samples=n_samples)
    stspRes=sts.query(**par,n_samples=n_samples,parallel=True)
    rsRes=rs.query(**par,n_samples=n_samples)
    dump(veRes,stsRes,stspRes,rsRes,par,n_samples,outPath)


if __name__ == "__main__":
    main()
