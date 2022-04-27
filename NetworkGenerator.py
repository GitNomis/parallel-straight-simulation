from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import truncnorm
import numpy as np

from utils import displayNetwork

folder = f'networks/'
states= ['Mono','Di','Tri','Tetra','Penta','Hexa','Hepta','Octa']
arities=['Unaray','Binary','Ternary','Quaternary','Quinary','Senary','Septenary','Octonary']

def generateCPD(var,var_card,evidence=[],evidence_card=[]):
    values=np.random.rand(var_card,np.product(evidence_card).astype(np.int64))
    values/=np.sum(values,0)
    state_names={}
    for v,c in zip([var]+evidence,[var_card]+evidence_card):
        state_names[v]=[state+arities[c-1] for state in states[:c] ]
    return TabularCPD(  variable=var,
                        variable_card=var_card,
                        values=values,
                        evidence=evidence,
                        evidence_card=evidence_card,
                        state_names=state_names)

def diamond(n,p):
    nodes = ['V'+str(i) for i in range(n)]
    edgeMatrix = (np.random.rand(n-1,n)<p).astype(np.int8)
    edges = [(nodes[i],nodes[j]) for i in range(n) for j in range(i+1,n) if edgeMatrix[i][j]]
    cards = [np.random.choice(range(1,9)) for _ in range(n)]
    model = BayesianNetwork(edges)
    for i in range(n):
        evidence = [nodes[j] for j in range(i) if edgeMatrix[j][i]]
        evidence_card = [cards[j] for j in range(i) if edgeMatrix[j][i]]
        model.add_cpds(generateCPD(nodes[i],cards[i],evidence,evidence_card))
    model.check_model()
    return model

def parent(n,inverse=False):
    nodes = ['V'+str(i) for i in range(n)]
    edges = [(node,nodes[0]) if inverse else (nodes[0],node) for node in nodes[1:]]
    cards = [np.random.choice(range(1,9)) for i in range(n)]
    model = BayesianNetwork(edges)
    if inverse:
        model.add_cpds(generateCPD(nodes[0],cards[0],nodes[1:],cards[1:]))
        for i in range(1,n):
            model.add_cpds(generateCPD(nodes[i],cards[i])) 
    else:
        model.add_cpds(generateCPD(nodes[0],cards[0]))
        for i in range(1,n):
            model.add_cpds(generateCPD(nodes[i],cards[i],[nodes[0]],[cards[0]]))    

    model.check_model()        
    return model


def chain(n):
    nodes = ['V'+str(i) for i in range(n)]
    cards = [np.random.choice(range(1,9)) for i in range(n)]
    model=BayesianNetwork(zip(nodes,nodes[1:]))
    model.add_cpds(generateCPD(nodes[0],cards[0]))
    for i in range(1,n):
        model.add_cpds(generateCPD(nodes[i],cards[i],[nodes[i-1]],[cards[i-1]]))
    model.check_model()    
    return model    

def main():
    n= 10
    inverse = False
    p=1
    model=parent(10,True)#diamond(n,p)
    displayNetwork(model)
    #file= f"chains/basic{n}"
    #file= f"parents/basic{n}{('_inverse' if inverse else '')}"
    file= f"diamonds/basic{n}_{p}"
    #model.save(folder+file+".bif")

if __name__ == "__main__":
    main()
