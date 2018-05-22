"""
Implement the sum-product algorithm for computing the mean matrix of z:
- input: theta (plus a and c)
- output: the mean matrix M (upper triangular form)
NB: we track the message dictionary throughout successive thetas
"""
import operator
import numpy as np
from numpy import exp, log
from scipy.special import gammaln
from . import config as cf
from . import utils as ut

### Counter of bad convergence
count_warning = 0

def get_graph(theta, threshold=0):
    """Compute the undirected graph induced by theta,
    possibly after applying a threshold.
    Returns the lists of nodes and edges."""
    G = np.size(theta[0])
    nodes = list(range(G))
    edges = []
    for i in range(G):
        for j in range(i+1,G):
            if (np.abs(theta[i,j]) > threshold):
                edges.append((i,j))
    return nodes, edges

def get_neighbors(nodes, edges):
    """Return the dictionary {node: neighbors} corresponding to the
    undirected graph defined by (nodes, edges). In the key-value pair
    (node, neighbors), neighbors is the set of node's neighbors."""
    neighbors = {node: set() for node in nodes}
    for (i,j) in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors

def update(message, edges, c):
    """Update the message dictionary given a list of edges,
    reusing previous messages when available."""
    for (i,j) in set(message.keys()) - set(edges):
        message.pop((i,j))
    for (i,j) in edges:
        message[i,j] = message.get((i,j), np.ones(c[j]+1))
        message[j,i] = message.get((j,i), np.ones(c[i]+1))
    return None

def stats(edges, a, theta, c):
    """Compute the sufficient statistics of the graphical model."""
    G = np.size(c)
    b = [ut.binomln(v) for v in c]
    z, psi = {}, {}
    for i in range(G):
        z[i] = np.arange(c[i]+1)
        l = a[0,i] + a[1,i]*z[i]
        psi[i] = exp(gammaln(l) - l*log(a[2,i]) + theta[i,i]*z[i] + b[i])
    for (i,j) in edges:
        psi[i,j] = exp(theta[i,j]*np.outer(z[i], z[j]))
        psi[j,i] = psi[i,j].T
    return z, psi

def state_pairs(nodes):
    """Return the set of tuples (i,j) with i < j in nodes."""
    spairs = set()
    for i in nodes:
        for j in nodes:
            if i < j: spairs.add((i,j))
    return spairs

def sumproduct(message, z, psi, neighbors):
    """Run the sum-product algorithm to compute messages.
    This function modifies the message dictionary in place."""
    global count_warning
    tol = cf.sp_tol
    iter_max = cf.sp_iter_max
    ### Initialization
    count, v = 0, tol+1
    message0 = {(i,j): message[i,j].copy() for (i,j) in message}
    ### The main loop
    while ((v > tol) & (count < iter_max)):
        ### Update all messages in parallel
        for (i,j) in message:
            n = neighbors[i] - {j}
            msgin = np.prod([message0[k,i] for k in n], axis=0)
            m = np.sum(psi[j,i] * psi[i] * msgin, axis=1)
            ### Do not forget to normalize
            message[i,j] = m/np.sum(m)
        ### Update the cache copy
        v = 0
        for (i,j) in message:
            s = np.max(np.abs(message[i,j] - message0[i,j]))
            v = np.max([v,s])
            message0[i,j][:] = message[i,j]
        # print(v)
        count += 1
    # print('SP done in {} iterations'.format(count))
    if (count == iter_max):
        count_warning += 1
        # msg = 'Warning: bad convergence of SP '
        # msg += '({} edges)'.format(int(len(message)/2))
        # print(msg)
    return None

def m0(i, j, neighbors, message, z, psi, theta, c):
    """Compute E[zi*zj] using the message dictionary (order 1).
    Neglects paths of length > 1 between i and j.
    Exact if i and j are neighbors and theta is a tree."""
    ni, nj = neighbors[i] - {j}, neighbors[j] - {i}
    mi = np.prod([message[k,i] for k in ni], axis=0)
    mj = np.prod([message[k,j] for k in nj], axis=0)
    Z = np.outer(z[i], z[j])
    P = np.outer(psi[i]*mi, psi[j]*mj) * exp(theta[i,j]*Z)
    P = P/np.sum(P)
    return np.sum(Z*P)

def m1(i, j, neighbors, message, psi, theta, c):
    """Compute E[zi*zj] using the message dictionary (order 2).
    Neglects paths of length > 2 between i and j.
    This is slower but more accurate than m0."""
    common = neighbors[i] & neighbors[j]
    ### Control of the clique size
    if len(common) > cf.sp_cmax-2:
        # print('Warning: clique with > {} nodes...'.format(cf.sp_cmax))
        d = {t:np.abs(theta[t,i])+np.abs(theta[t,j]) for t in common}
        score = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        common = {node[0] for node in score[0:(cf.sp_cmax-2)]}
    clique = common | {i,j}
    # print('Clique = {} ({} - {})'.format(clique, i, j))
    G, mu = np.size(c), {}
    for t in clique:
        n = neighbors[t] - clique
        mu[t] = psi[t] * np.prod([message[s,t] for s in n], axis=0)
    f = {t:[0] for t in set(range(G)) - clique}
    states = ut.state_vector(c, fixed=f)
    pairs = state_pairs(clique)
    Z = np.array([z[i]*z[j] for z in states])
    P = np.zeros(len(states))
    for k, z in enumerate(states):
        r = np.sum([theta[s,t]*z[s]*z[t] for (s,t) in pairs])
        P[k] = np.prod([mu[t][z[t]] for t in clique]) * exp(r)
    P = P/np.sum(P)
    return np.sum(Z*P)

def meanz(message, a, theta, c):
    """Estimate the mean matrix using the sum-product algorithm.
    This function modifies the message dictionnary in place.
    NB: the output might be a rough approximation."""
    s = cf.sp_threshold
    ### Initialization
    nodes, edges = get_graph(theta, threshold=s)
    neighbors = get_neighbors(nodes, edges)
    z, psi = stats(edges, a, theta, c)
    ### Compute the messages given theta
    update(message, edges, c)
    sumproduct(message, z, psi, neighbors)
    ### Compute the mean matrix of z
    G = np.size(c)
    M = np.zeros((G,G))
    ### Classic method for M[i,i] - exact if theta is a tree
    for i in range(G):
        ni = neighbors[i]
        p = psi[i] * np.prod([message[k,i] for k in ni], axis=0)
        p = p/np.sum(p)
        M[i,i] = np.sum(z[i]*p)
    ### Custom method for M[i,j]
    for i in range(G):
        for j in range(i+1,G):
            ### Order one: only check for common neighbors
            common = neighbors[i] & neighbors[j]
            if len(common) == 0:
                if theta[i,j] == 0: M[i,j] = M[i,i] * M[j,j]
                else: M[i,j] = m0(i, j, neighbors, message, z, psi, theta, c)
            else: M[i,j] = m1(i, j, neighbors, message, psi, theta, c)
            ### Do not forget to symmetrize
            M[j,i] = M[i,j]
    return M


# ### MANIPULATION DE GRAPHES
# def symmetrize(graph):
#     """Add arcs so that the graph becomes the
#     undirected version of the original graph.
#     The input is modified in place."""
#     for f, s in graph.items():
#         for i in s:
#             if f not in graph[i]: graph[i].append(f)

# def find_tree(nodes, edges, root):
#     """Build the "shortest" directed tree from an undirected graph,
#     starting from a fixed root. Some edges are discarded if the
#     input graph is not triangulated."""
#     def tree(nodes, edges, root):
#         graph = {root: []}
#         for (i,j) in edges.copy():
#             if root == i and j in nodes:
#                 graph[root] += [j]
#                 nodes.remove(j)
#                 edges.remove((i,j))
#             elif root == j and i in nodes:
#                 graph[root] += [i]
#                 nodes.remove(i)
#                 edges.remove((i,j))
#         for subroot in graph[root]:
#             subgraph = tree(nodes, edges, subroot)
#             subnodes = graph.keys()
#             for f, s in subgraph.items():
#                 if f in subnodes: graph[f] += s
#                 else: graph[f] = s
#         return graph
#     ### The main function is wrapped to avoid side effects
#     return tree(nodes.copy(), edges.copy(), root)

# def find_path(graph, start, end, path=[]):
#     """Find a path in a directed graph. If the graph is a tree
#     with root equal to start, this path exists and is unique."""
#     path = path + [start]
#     if start == end:
#         return path
#     if not start in graph.keys():
#         return None
#     for node in graph[start]:
#         if node not in path:
#             newpath = find_path(graph, node, end, path)
#             if newpath: return newpath
#     return None

# def find_shortest_path(graph, start, end, path=[]):
#         path = path + [start]
#         if start == end:
#             return path
#         if not start in graph.keys():
#             return None
#         shortest = None
#         for node in graph[start]:
#             if node not in path:
#                 newpath = find_shortest_path(graph, node, end, path)
#                 if newpath:
#                     if not shortest or len(newpath) < len(shortest):
#                         shortest = newpath
#         return shortest


# ### TESTS
# G = 100
# theta = np.zeros((G,G))
# for i in range(G):
#     for j in range(i+1,G):
#         # theta[i,j] = np.random.randint(-1,2)
#         theta[i,j] = 0.05*np.random.randn()
# nodes, edges = get_graph(theta, 0.1)
# # print(nodes, edges)

# ### Arbre pour les tests
# # nodes, edges = get_graph(theta_arbre)
# # print(nodes, edges)

# graph1 = find_tree(nodes, edges, 0) # Choose the root randomly
# # graph2 = find_tree(nodes, edges, 3)
# # print(graph1)
# # print(graph2)
# symmetrize(graph1)
# # symmetrize(graph2)
# # print(graph1)
# # print(graph2)
# # print(neighbours(edges, 5))
# print(find_path(graph1, 0, 5))
# # print(find_shortest_path(graph1, 5, 0))