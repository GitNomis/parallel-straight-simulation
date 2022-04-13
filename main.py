from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VE
from pgmpy.inference import ApproxInference as RS
from StraightSimulation import StraightSimulation as StS
from utils import *


file = 'earthquake'
inPath = f'networks/{file}.bif'
outPath = f'output/{file}.out'
args = {'variables':[ 'MaryCalls','JohnCalls'], 'evidence': {'Earthquake':'True','Burglary':'True'}}
#args = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
#args = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
n_samples = 1000


def main():
    network = BN.load(inPath)
    #displayNetwork(network)
    ve = VE(network)
    sts = StS(network)
    rs = RS(network)
    veRes=ve.query(**args)
    stsRes=sts.query(**args,n_samples=n_samples)
    stspRes=sts.query(**args,n_samples=n_samples,parallel=True)
    rsRes=rs.query(**args,n_samples=n_samples)
    test(veRes,stsRes,stspRes,rsRes,args,n_samples,outPath)
    # args['n_samples']=n_samples
    # multiRun(veRes,rs,args,100)
    # multiRun(veRes,sts,args,100)
    # args['parallel']=True
    # multiRun(veRes,sts,args,100)


if __name__ == "__main__":
    main()
