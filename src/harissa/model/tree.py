"""
Generate random trees
"""
import numpy as np

def random_step(state, a):
    """
    Make one step of the random walk on the weighted graph defined by a.
    NB: here we construct an in-tree so all directions are reversed.
    """
    p = a[:,state]/np.sum(a[:,state])
    return np.dot(np.arange(p.size), np.random.multinomial(1, p))

def loop_erasure(path):
    """
    Compute the loop erasure of a given path.
    """
    if path[0] == path[-1]: return [path[0]]
    else: i = np.max(np.arange(len(path))*(np.array(path)==path[0]))
    if path[i+1] == path[-1]: return [path[0], path[i+1]]
    else: return [path[0]] + loop_erasure(path[i+1:])

def random_tree(a):
    """
    Generate a random spanning tree rooted in node 0 from the uniform
    distribution with weights given by matrix a (using Wilson's method).
    """
    n = a[0].size
    tree = [[] for i in range(n)]
    v = {0} # Vertices of the tree
    r = list(range(1,n)) # Remaining vertices
    while len(r) > 0:
        state = r[0]
        path = [state]
        # compute a random path that reaches the current tree
        while path[-1] not in v:
            state = random_step(path[-1], a)
            path.append(state)
        path = loop_erasure(path)
        # Append the loop-erased path to the current tree
        for i in range(len(path)-1):
            v.add(path[i])
            r.remove(path[i])
            tree[path[i+1]].append(path[i])
    for i in range(n): tree[i].sort()
    return tuple([tuple(tree[i]) for i in range(n)])

# Main function
def tree(n_genes, weight=None):
    """
    Generate a random tree-like network model.
    A tree with root 0 is sampled from the ‘weighted-uniform’ distribution,
    where weight[i,j] is the probability weight of link (i) -> (j).
    """
    G = n_genes + 1
    if weight is not None:
        if weight.shape != (G,G):
            raise ValueError('Weight must be n_genes+1 by n_genes+1')
    else: weight = np.ones((G,G))
    # Enforcing the proper structure
    weight[:,0] = 0
    weight = weight - np.diag(np.diag(weight))
    # Generate the network
    tree = random_tree(weight)
    basal = np.zeros(G)
    inter = np.zeros((G,G))
    basal[1:] = -5
    for i, targets in enumerate(tree):
        for j in targets:
            inter[i,j] = 10
    return basal, inter


# Tests
if __name__ == '__main__':
    basal, inter = tree(5)
    print(basal)
    print(inter)
