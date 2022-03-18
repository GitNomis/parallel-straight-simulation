import pgmpy
import daft
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork as BN
from pgmpy.inference import VariableElimination as VE
from pgmpy.inference import ApproxInference as AI
from StraightSimulation import StraightSimulation as SS


file = 'networks/earthquake.bif'


def main():
    par = {'variables':['JohnCalls', 'MaryCalls'], 'evidence': {'Burglary': 'True'}}
    network = BN.load(file)
    # network.to_daft(node_pos='spring').render()
    # plt.show()
    ve = VE(network)
    ai = AI(network)
    ss = SS(network)
    #print(ve.query(**par))
    print(ss.query(**par))
    print(ai.query(**par,n_samples=10000))


if __name__ == "__main__":
    main()
