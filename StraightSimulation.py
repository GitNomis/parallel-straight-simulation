from collections import defaultdict
from pgmpy.sampling import BayesianModelInference
from pgmpy.factors.discrete import DiscreteFactor
import numpy as np


class StraightSimulation(BayesianModelInference):

    def query(self, variables, n_samples=10000, evidence=None):
        # Init
        # self._initialize_structures()
        workModel = self.model.copy()
        simVars = [var for var in self.topological_order if var not in evidence]

        nameToIndex = {var: i for i, var in enumerate(self.topological_order)}
        valueToInt = {var: {val: i for i, val in values.items()}
                      for var, values in self.state_names_map.items()}
        # Fix evidence
        cpds = [cpd.to_factor() for cpd in workModel.get_cpds()]

        factors = defaultdict(list)
        for cpd in cpds:
            for var in cpd.scope():
                factors[var].append(cpd)
            observedEvidence = [
                (var, val) for var, val in evidence.items() if var in cpd.scope()]
            if observedEvidence:
                cpd.reduce(observedEvidence, inplace=True)

        # Init state forward
        state = np.zeros(len(self.topological_order), dtype=int)
        for var, val in evidence.items():
            state[nameToIndex[var]] = valueToInt[var][val]

        results = np.zeros([self.cardinality[v] for v in variables])
        # future better start state (forward sample)
        # set of indecies for updating states
        # Loop N times
        for i in range(n_samples):

            # Loop vars
            for var in simVars:
                # markov blanket
                p = np.ones(self.cardinality[var])
                # calc P
                for cpd in factors[var]:
                    reduceEvidence = [(evi, self.state_names_map[evi][state[nameToIndex[evi]]])
                                      for evi in simVars if evi in cpd.scope() and evi != var]
                    p *= cpd.reduce(reduceEvidence, inplace=False).values
                # sample
                state[nameToIndex[var]] = np.random.choice(
                    p.size, p=(p/np.sum(p)))

            # update state
            subState = state[[nameToIndex[v] for v in variables]]
            results[tuple(subState)] += 1
            # note result

        # normalize with N
        results /= n_samples

        return DiscreteFactor(variables, results.shape, results, {k: v for k, v in workModel.states.items() if k in variables})
