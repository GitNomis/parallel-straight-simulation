import pgmpy
import daft
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VE
from pgmpy.inference import ApproxInference as RS
from StraightSimulation import StraightSimulation as StS


file = 'networks/earthquake_test.bif'


def main():
    par = {'variables':['Earthquake', 'Burglary'], 'evidence': {'MaryCalls':'True','JohnCalls':'True'}}
    #par = {'variables':['AppOK', 'DataFile'], 'evidence': {'Problem4':'Yes','TTOK':'Yes'}}
    #par = {'variables':['alcoholism', 'ChHepatitis'], 'evidence': {'diabetes':'present','pain_ruq':'present'}}
    network = BN.load(file)
    # network.to_daft(node_pos='spring').render()
    # plt.show()
    ve = VE(network)
    sts = StS(network)
    rs = RS(network)
    veRes=ve.query(**par)
    stsRes=sts.query(**par,n_samples=10000)
    stspRes=sts.query(**par,n_samples=10000,parallel=True)
    rsRes=rs.query(**par,n_samples=10000)
    print("Variable Elimination:\n",veRes)
    print("Straight Simluation:\n",stsRes)
    print("Parallel Straight Simluation:\n",stspRes)
    print("Rejection Sampling:\n",rsRes)


if __name__ == "__main__":
    main()
