from collections import defaultdict
from pgmpy.sampling import BayesianModelInference
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.global_vars import SHOW_PROGRESS
import numpy as np
from tqdm import trange


class Processor():
    def __init__(self, var, factors):
        """
        Initilizes the Processor class, which is responsible for simulating a specific variable.

        Args:
            var (str): The name of the variable to be simulated.
            factors (list): The list of factors including the variable `var`.
        """

        self.var = var
        self.factors = factors

    def sample(self, state, output):
        """
        Calculates the probability distribution of the variable corresponding to this Processor and samples a new value.

        Args:
            state (numpy.array): The current state to consider for sampling.
            output (numpy.array): The output state to write the new sample to.

        Returns:
            p,sample (numpy.array,int): The probability distribution and sampled value.
        """

        p = 1
        # Calculate probability distribution
        for cpd in self.factors:
            evidence = [(evi, cpd.get_state_names(evi, state[evi]))
                        for evi in cpd.scope() if evi != self.var]
            p *= cpd.reduce(evidence, inplace=False).values

        # Normalize
        p /= np.sum(p)
        # Sample
        sample = np.random.choice(p.size, p=p)
        output[self.var] = sample

        return p, sample


class StraightSimulation(BayesianModelInference):

    def query(self, variables, n_samples=10000, evidence={}, parallel=False, show_progress=True):
        """
        Approximates the posterior distribution of a given query using the Straight Simulation method.
        Straight Simulation can be run completly parallelised, with different convergence properties.

        Args:
            variables (list): Variables to query.
            n_samples (int, optional): Number of samples to generate. Defaults to 10000.
            evidence (dict, optional): Evidence of the query. Defaults to {}.
            parallel (bool, optional): Whether to run parallel Straight Simulation. Defaults to False.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            DiscreteFactor: The discrete factor containing the approximated distribution for the given query
        """

        # Simulation Order (#TODO different simulation order)
        nodes = [
            var for var in self.topological_order if var not in evidence]

        # Create states and results
        state = np.array(0, dtype=[(var, np.int8) for var in nodes])
        output = state if parallel else np.array(
            0, dtype=[(var, np.int8) for var in nodes])

        results = np.zeros([self.cardinality[v] for v in variables])

        # Prepare factors
        factors = defaultdict(list)
        for node in self.topological_order:
            factor = self.model.get_cpds(node).to_factor()
            
            # Fix evidence
            relevantEvidence = [
                (var, val) for var, val in evidence.items() if var in factor.scope()]
            if relevantEvidence:
                factor.reduce(relevantEvidence, inplace=True)
            for var in factor.scope():
                factors[var].append(factor)

            # Forward sample initial state
            if node not in evidence:
                relevantEvidence = [(evi, factor.get_state_names(evi, state[evi]))
                                    for evi in factor.scope() if evi != node]
                p = factor.reduce(relevantEvidence, inplace=False).values
                # Sample
                state[node] = np.random.choice(p.size, p=p)

        # Create Processor object for each variable
        Processors = [Processor(var, factors[var]) for var in nodes]

        # Progress bar
        if show_progress and SHOW_PROGRESS:
            pbar = trange(n_samples)
            pbar.set_description(f"Generating samples")
        else:
            pbar = range(n_samples)

        # Loop N times
        for i in pbar:

            # Loop variables
            for var in Processors:
                # Update current state or next state
                p, value = var.sample(state, output)

            # Note result (#TODO test saving probabilities)
            results[tuple(output[variables])] += 1

            # Swap states for next iteration
            if not parallel:
                help = state
                state = output
                output = help

        # normalize with N
        results /= n_samples

        return DiscreteFactor(variables, results.shape, results, {v: self.model.states[v] for v in variables})
