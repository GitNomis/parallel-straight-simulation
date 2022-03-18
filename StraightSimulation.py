from collections import defaultdict
from pgmpy.sampling import BayesianModelInference
import pandas as pd
import numpy as np
# from pgmpy.factors import

"""
print(self.model)
BayesianNetwork named 'unknown' with 5 nodes and 4 edges
print(self.variables)
['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']
print(self.cardinality)
{'Burglary': 2, 'Earthquake': 2, 'Alarm': 2, 'JohnCalls': 2, 'MaryCalls': 2}
print(self.factors)
defaultdict(<class 'list'>, {'Burglary': [<DiscreteFactor representing phi(Burglary:2) at 0x15bb3f4a988>, <DiscreteFactor representing phi(Alarm:2, Burglary:2, Earthquake:2) 
at 0x15bb3f4ae88>], 'Earthquake': [<DiscreteFactor representing phi(Earthquake:2) at 0x15bb3f4ae08>, <DiscreteFactor representing phi(Alarm:2, Burglary:2, Earthquake:2) at 0x15bb3f4ae88>], 'Alarm': [<DiscreteFactor representing phi(Alarm:2, Burglary:2, Earthquake:2) at 0x15bb3f4ae88>, <DiscreteFactor representing phi(JohnCalls:2, Alarm:2) at 0x15bb3f4af88>, <DiscreteFactor representing phi(MaryCalls:2, Alarm:2) at 0x15bb3f55dc8>], 'JohnCalls': [<DiscreteFactor representing phi(JohnCalls:2, Alarm:2) at 0x15bb3f4af88>], 'MaryCalls': [<DiscreteFactor representing phi(MaryCalls:2, Alarm:2) at 0x15bb3f55dc8>]})
print(self.state_names_map)
{'Burglary': {0: 'True', 1: 'False'}, 'Earthquake': {0: 'True', 1: 'False'}, 'Alarm': {0: 'True', 1: 'False'}, 'JohnCalls': {0: 'True', 1: 'False'}, 'MaryCalls': {0: 'True', 
1: 'False'}}
self.model.get_cpds(var[0])
+------------------+-------------+--------------+
| Alarm            | Alarm(True) | Alarm(False) |
+------------------+-------------+--------------+
| JohnCalls(True)  | 0.9         | 0.05         |
+------------------+-------------+--------------+
| JohnCalls(False) | 0.1         | 0.95         |
+------------------+-------------+--------------+
"""


class StraightSimulation(BayesianModelInference):
    def query(self, variables, n_samples=10000, evidence=None):
        # Init
        # self._initialize_structures()
        workModel = self.model.copy()
        simVars= [i for i,var in enumerate(self.topological_order) if var not in evidence]
        nameToIndex = {var: i for i, var in enumerate(self.topological_order)}
        valueToInt = {var: {val: i for i, val in values.items()}
                      for var, values in self.state_names_map.items()}
        # Fix evidence (elim vars?)
        """
        factors = self.factors

        for e in evidence:
            for ef in factors[e]:
                ref = ef.reduce([(e,evidence[e])],inplace=False)
                for v in ref.variables:
                    factors[v].remove(ef)
                    factors[v].append(ref)
            del factors[e]        
        """
        cpds = workModel.get_cpds()
        
        for cpd in cpds:
            observedEvidence = [
                (var, val) for var, val in evidence.items() if var in cpd.get_evidence()]
            if observedEvidence:
                print(cpd.scope())
                cpd.reduce(observedEvidence, inplace=True)
                print(cpd.scope())


        # Init state forward
        state = np.zeros(len(self.topological_order))
        for var, val in evidence.items():
            state[nameToIndex[var]] = valueToInt[var][val]

        # future better start state (forward sample)
            
        # Loop N times
        for i in range(n_samples):

        # Loop vars
            for var in simVars:

        # markov blanket

        # calc P

        # sample

        # update state

        # note result

        # normalize with N

        return self.model.get_cpds(variables[0])
