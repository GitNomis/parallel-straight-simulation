import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from tqdm import tqdm, trange

from utils import displayNetwork

folder = f'networks/'
states = ['Mono', 'Di', 'Tri', 'Tetra', 'Penta', 'Hexa', 'Hepta', 'Octa']
arities = ['Unaray', 'Binary', 'Ternary', 'Quaternary',
           'Quinary', 'Senary', 'Septenary', 'Octonary']
card_probs = [0.02, 0.45, 0.25, 0.10, 0.08, 0.05, 0.03, 0.02]


def generateCPD(var, var_card, evidence=np.array([]), evidence_card=np.array([])):
    values = np.random.rand(var_card, np.product(
        evidence_card, dtype=np.int64))
    values /= np.sum(values, 0)
    state_names = {}
    for v, c in zip([var]+list(evidence), [var_card]+list(evidence_card)):
        state_names[v] = [state+arities[c-1] for state in states[:c]]
    return TabularCPD(variable=var,
                      variable_card=var_card,
                      values=values,
                      evidence=evidence,
                      evidence_card=evidence_card,
                      state_names=state_names)


def generateCards(n):
    return np.random.choice(range(1, 9), n, p=card_probs)


def diamond(n, p):
    nodes = np.array(['V'+str(i) for i in range(n)])
    edgeMatrix = np.triu(np.random.rand(n, n) < p, 1)
    edges = [(nodes[i], nodes[j]) for i in range(n)
             for j in range(i+1, n) if edgeMatrix[i][j]]
    cards = generateCards(n)
    model = BayesianNetwork(edges)
    model.add_nodes_from(nodes)
    for i in range(n):
        evidence = nodes[edgeMatrix[:, i]]
        evidence_card = cards[edgeMatrix[:, i]]
        model.add_cpds(generateCPD(
            nodes[i], cards[i], evidence, evidence_card))
    model.check_model()
    return model


def parent(n, inverse=False):
    nodes = np.array(['V'+str(i) for i in range(n)])
    edges = [(node, nodes[0]) if inverse else (nodes[0], node)
             for node in nodes[1:]]
    cards = generateCards(n)
    model = BayesianNetwork(edges)
    if inverse:
        model.add_cpds(generateCPD(nodes[0], cards[0], nodes[1:], cards[1:]))
        for i in range(1, n):
            model.add_cpds(generateCPD(nodes[i], cards[i]))
    else:
        model.add_cpds(generateCPD(nodes[0], cards[0]))
        for i in range(1, n):
            model.add_cpds(generateCPD(
                nodes[i], cards[i], [nodes[0]], [cards[0]]))

    model.check_model()
    return model


def chain(n):
    nodes = np.array(['V'+str(i) for i in range(n)])
    edges = zip(nodes, nodes[1:])
    cards = generateCards(n)
    model = BayesianNetwork(edges)
    model.add_cpds(generateCPD(nodes[0], cards[0]))
    for i in range(1, n):
        model.add_cpds(generateCPD(
            nodes[i], cards[i], [nodes[i-1]], [cards[i-1]]))
    model.check_model()
    return model


def chromatic_number(network, show_progress=False):
    result = dict()
    pbar = trange(1, 1+len(network.nodes), leave=None,
                  disable=not show_progress)
    for i in pbar:
        pbar.set_description(f"Attempting to color using {i} colors")
        result = try_color(network, set(network.nodes),
                           dict(), i, show_progress)
        if result:
            return i, result


def try_color(network, variables, colors, max_colors, show_progress):
    if show_progress:
        show_progress -= 1
    if not variables:
        return colors
    variable = variables.pop()

    new_color = min(max(colors.values(), default=-1)+1, max_colors-1)
    color_options = {*colors.values(), new_color}
    color_options -= set(colors.get(var, None)
                         for var in network.adj[variable])

    pbar = tqdm(color_options, leave=None, disable=not show_progress)
    for col in pbar:
        pbar.set_description(f"Trying {variable} = {col}")
        colors[variable] = col
        result = try_color(network, variables.copy(), colors.copy(),
                           max_colors, show_progress)
        if result:
            return result
    return dict()


def main():
    network_funtion = diamond
    name = "basic"
    if network_funtion == chain:
        n = 50
        file = f"chains/{name}{n}"
        args = {'n': n}
    elif network_funtion == parent:
        n = 4
        inverse = False
        file = f"parents/{name}{n}{('_inverse' if inverse else '')}"
        args = {'n': n, 'inverse': inverse}
    elif network_funtion == diamond:
        n = 10
        p = 0.3
        file = f"diamonds/{name}{n}_{p}"
        args = {'n': n, 'p': p}

    model = network_funtion(**args)
    displayNetwork(model)
    model.save(folder+file+".bif", filetype='bif')


if __name__ == "__main__":
    main()
