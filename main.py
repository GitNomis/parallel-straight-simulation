import pgmpy
import daft
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VE
from pgmpy.inference import ApproxInference as RS
from StraightSimulation import StraightSimulation as StS


file = 'networks/earthquake.bif'


def main():
    par = {'variables':['Earthquake', 'Burglary'], 'evidence': {'JohnCalls':'True','MaryCalls':'True'}}
    network = BN.load(file)
    # network.to_daft(node_pos='spring').render()
    # plt.show()
    ve = VE(network)
    sts = StS(network)
    rs = RS(network)
    print("Variable Elimination:\n",ve.query(**par))
    print("Straight Simluation:\n",sts.query(**par,n_samples=100000))
    print("Rejection Sampling:\n",rs.query(**par,n_samples=100000))


if __name__ == "__main__":
    main()
