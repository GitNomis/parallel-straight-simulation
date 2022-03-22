from collections import defaultdict
from pgmpy.sampling import BayesianModelInference
from pgmpy.factors.discrete import DiscreteFactor
import numpy as np


class ProcessorVariable():
    def __init__(self, factors, var, state_names_map,par):
        self.state_names_map = state_names_map
        self.factors = factors
        self.var = var
        self.par = par

    def getValue(self):
        pass

    def sample(self,state,newState=None):
        p = np.ones(len(self.state_names_map))
        # calc P
        for cpd in self.factors:
            reduceEvidence = [(evi, state[evi])
                              for evi in cpd.scope() if evi != self.var]
            p *= cpd.reduce(reduceEvidence, inplace=False).values
        # sample
        p /= np.sum(p)
        sample = np.random.choice(p.size, p=p)
        if self.par:
            newState[self.var]= self.state_names_map[sample]
        else:    
            state[self.var] = self.state_names_map[sample]
        return p, sample


class StraightSimulation(BayesianModelInference):

    def query(self, variables, n_samples=10000,par=False, evidence={}):
        # Init
        workModel = self.model.copy()

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
        # future better start state (forward sample)
        state = {var: self.state_names_map[var][0]
                 for var in self.topological_order if var not in evidence}
        newState =  {var: self.state_names_map[var][0]
                 for var in self.topological_order if var not in evidence}       
        simVars = []
        for var in state:
            simVars.append(ProcessorVariable(
                factors[var], var, self.state_names_map[var],par))

        results = np.zeros([self.cardinality[v] for v in variables])

        # Loop N times
        for i in range(n_samples):
            # Loop vars
            for var in simVars:
                p, value = var.sample(state,newState)
            if par:
                inte=state
                state=newState
                newState=inte    

            # update state
            subState = [valueToInt[v][state[v]] for v in variables]
            results[tuple(subState)] += 1
            # note result

        # normalize with N
        results /= n_samples

        return DiscreteFactor(variables, results.shape, results, {v: workModel.states[v] for v in variables})
