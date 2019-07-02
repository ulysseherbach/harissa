"""
Plotting networks
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc, Polygon
from matplotlib.patches import ArrowStyle

activ = '#96D255'
inhib = '#E97356'

def forces(p, e, width, height, root=False):
    """
    Compute the vector of forces for vertices of a given graph.
    Based on the spring embedder of Fruchterman & Reingold (1991).
    """
    n = p[:,0].size # Number of vertices
    l = 0.6 # Ideal length of an edge
    f1 = np.zeros((n,2))
    f2 = np.zeros((n,2))
    f3 = np.zeros((n,2))
    # 0. Repulsive force at boundaries
    f1[:] = (l**2) * (1/p - 1/(np.array([width,height]) - p))
    # 1. Repulsive force between all node pairs
    for i in range(n):
        for j in range(i+1,n):
            u = p[i] - p[j]
            d = np.sqrt(np.sum(u**2))
            f1[i] += ((l/d)**2) * u
            f1[j] -= ((l/d)**2) * u
    # 2. Attractive force between adjacent nodes
    for i, j in e:
        if i != j:
            u = p[j] - p[i]
            d = np.sqrt(np.sum(u**2))
            f2[i] += (d/l) * u
            f2[j] -= (d/l) * u
            # Add some gravity
            if root:
                g = 0.1 * n
                v = np.array([u[1],-u[0]])/d
                f2[j] += (g/d) * u[0] * v
    # 3. Add rigidity for edge length
    alpha = 10 * n
    for i, j in e:
        if i != j:
            u = p[j] - p[i]
            d = np.sqrt(np.sum(u**2))
            f3[i] += alpha * (d-l) * ((l/d)**2) * u
            f3[j] -= alpha * (d-l) * ((l/d)**2) * u
    # Resulting force
    f = f1 + f2 + f3
    # c = np.max(np.sqrt(np.sum(f**2, axis=1)))
    # if c > 2: f -= f3
    if root: f[0] = 0
    return f/(n*np.sqrt(width**2 + height**2))

def graph_layout(v, e, width, height, tol=None, root=False, verb=False):
    """
    Compute a relevant graph layout by minimizing an energy.
    The output is a list of tuples [(x0,y0),(x1,y1),...] where
    xv and yv are coordinates of vertex v, both within [0,1].
    """
    if tol is None: tol = 1e-3
    max_iter = 10000
    n = len(v)
    xmin, ymin = 1/n, 1/n
    xmax, ymax = (1 - 1/n)*width, (1 - 1/n)*height
    # Initialization
    p = np.zeros((n,2))
    p[:,0] = np.random.uniform(xmin, xmax, n)
    p[:,1] = np.random.uniform(ymin, ymax, n)
    if root:
        p[0,0] = width/2
        p[0,1] = (1 - 1/(n+1))*height
    # Energy minimization
    delta = 1.5e-1
    k, c = 0, 0
    while (k == 0) or (k < max_iter and c > tol):
        f = forces(p, e, width, height, root=root)
        p = p + delta*f
        c = np.max(np.sqrt(np.sum(f**2, axis=1)))
        k += 1
        if verb: print('c = {}'.format(c))
    if (k == max_iter) and (c > tol): print('Warning: no convergence')
    elif verb: print('Converged in {} iterations'.format(k))
    return p


# Plotting functions
def node(k, ax, p, name, color=None, fontsize=None):
    if color is None: color = '#F2F2F2'
    if fontsize is None: fontsize = 8
    r = 0.1
    x, y = p[k,0], p[k,1]
    circle0 = plt.Circle((x, y), radius=r, color=color)
    circle0.zorder=2
    ax.add_artist(circle0)
    if k != 0:
        # circle0 = plt.Circle((x, y), radius=r, color=color, zorder=2)
        # ax.add_artist(circle0)
        circle1 = plt.Circle((x, y), radius=r, fill=False,
            color='lightgray', lw=1, zorder=3)
        ax.add_artist(circle1)
        ax.text(x, y - 0.007, name, color='gray', fontsize=fontsize, zorder=4,
            horizontalalignment='center', verticalalignment='center')
    else:
        # circle1 = plt.Circle((x, y), radius=r, fill=False,
            # color='darkgray', lw=1, zorder=3)
        circle1 = plt.Circle((x, y), radius=r, fill=False,
            color='lightgray', lw=1, zorder=3)
        ax.add_artist(circle1)

# def link(k1, k2, ax, p, weight):
#     l = 0.015
#     r = 0.125
#     u = p[k2] - p[k1]
#     u0 = u/np.sqrt(np.sum(u**2))
#     x0 = p[k1,0] + r*u0[0]
#     y0 = p[k1,1] + r*u0[1]
#     dx = u[0] - 2*r*u0[0]
#     dy = u[1] - 2*r*u0[1]
#     # Case 1: activation
#     if weight > 0:
#         arrow = FancyArrow(x0, y0, dx, dy, width=l, lw=0, color=activ,
#             length_includes_head=True, head_width=0.05, head_length=None)
#         arrow.zorder = 0
#         ax.add_artist(arrow)
#     # Case 2: inhibition
#     elif weight < 0:
#         arrow = FancyArrow(x0, y0, dx, dy, width=l, lw=0, color=inhib,
#             length_includes_head=True, head_width=0, head_length=0)
#         arrow.zorder = 0
#         ax.add_artist(arrow)
#         h_width = 0.07
#         h_height = l
#         x1, y1 = x0 + dx, y0 + dy
#         d = np.sqrt(dx**2 + dy**2)
#         u = np.array([dx, dy])/d
#         v = np.array([dy,-dx])/d
#         x2 = x1 + 0.2*h_height*u[0] + 0.5*h_width*v[0]
#         y2 = y1 + 0.2*h_height*u[1] + 0.5*h_width*v[1]
#         if v[0] > 0: angle = (180/np.pi)*np.arctan(v[1]/v[0]) - 180
#         elif v[0] < 0: angle = (180/np.pi)*np.arctan(v[1]/v[0])
#         else: angle = 90
#         head = Rectangle((x2,y2), h_width, h_height, angle=angle, fc=inhib)
#         head.zorder = 0
#         ax.add_artist(head)

def link(k1, k2, ax, p, weight, bend=0):
    # Node coordinates
    x1, y1 = p[k1,0], p[k1,1]
    x2, y2 = p[k2,0], p[k2,1]

    # Case 1: activation
    if weight > 0:
        style = ArrowStyle('Simple', tail_width=1.1,
            head_width=3.5, head_length=5)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=9, shrinkB=9, fc=activ, lw=0, zorder=0,
        connectionstyle='arc3,rad={}'.format(bend))
        ax.add_artist(arrow)

    # Case 2: inhibition
    if weight < 0:
        style = ArrowStyle('Simple', tail_width=1.1,
            head_width=0, head_length=1e-9)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=9, shrinkB=9, fc=inhib, lw=0, zorder=0,
        connectionstyle='arc3,rad={}'.format(bend))
        ax.add_artist(arrow)

        r = 0.125
        u = p[k2] - p[k1]
        u0 = u/np.sqrt(np.sum(u**2))
        x0 = p[k1,0] + r*u0[0]
        y0 = p[k1,1] + r*u0[1]
        dx = u[0] - 2*r*u0[0]
        dy = u[1] - 2*r*u0[1]

        h_width = 0.07
        h_height = 0.015
        x1, y1 = x0 + dx, y0 + dy
        d = np.sqrt(dx**2 + dy**2)
        u = np.array([dx, dy])/d
        v = np.array([dy,-dx])/d

        x = x1 + 0.2*h_height*u[0] + 0.5*h_width*v[0]
        y = y1 + 0.2*h_height*u[1] + 0.5*h_width*v[1]

        if v[0] > 0: angle = (180/np.pi)*np.arctan(v[1]/v[0]) - 180
        elif v[0] < 0: angle = (180/np.pi)*np.arctan(v[1]/v[0])
        else: angle = 90

        angle += np.tanh(bend) * 90
        theta = np.tanh(bend) * (np.pi/2)

        x3 = x2 + np.cos(theta)*(x-x2) - np.sin(theta)*(y-y2)
        y3 = y2 + np.sin(theta)*(x-x2) + np.cos(theta)*(y-y2)

        head = Rectangle((x3,y3), h_width, h_height,
            angle=angle, fc=inhib, zorder=0)
        ax.add_artist(head)

def link_auto(k, ax, p, weight, v=None):
    # Node coordinates
    x0, y0 = p[k,0], p[k,1]

    # Orientation
    if v is None:
        v = np.array([1,0])
        v = v/np.sqrt(np.sum(v**2))

    if v[0] > 0: angle0 = np.arcsin(v[1]) * (180/np.pi)
    else: angle0 = 180 - np.arcsin(v[1]) * (180/np.pi)

    angle1 = 58 + angle0
    angle2 = 300 + angle0
    theta = angle0 * (np.pi/180)

    x, y = x0 + 0.145, y0
    x1 = x0 + np.cos(theta)*(x-x0) - np.sin(theta)*(y-y0)
    y1 = y0 + np.sin(theta)*(x-x0) + np.cos(theta)*(y-y0)

    # Loop diameter
    d = 0.15

    # Case 1: activation
    if weight > 0:
        c = activ
        arc = Arc((x1,y1), d, d, angle=180, theta1=angle1, theta2=angle2-30,
        lw=1.1, color=c, zorder=0)
        ax.add_artist(arc)

        # Activation head
        head_width = 0.045
        head_length = 0.065

        angle = angle2 - 180 - 30
        theta = angle * (np.pi/180)

        triangle = np.array([
            [x1 + d/2 + head_width/2, y1 - head_length/4],
            [x1 + d/2, y1 + head_length * 3/4],
            [x1 + d/2 - head_width/2, y1 - head_length/4]])
        x, y = triangle[:,0].copy(), triangle[:,1].copy()
        triangle[:,0] = x1 + np.cos(theta)*(x-x1) - np.sin(theta)*(y-y1)
        triangle[:,1] = y1 + np.sin(theta)*(x-x1) + np.cos(theta)*(y-y1)
        triangle -= 0.03 * (triangle[0]-triangle[2])

        head = Polygon(triangle, fc=activ, zorder=0)
        ax.add_artist(head)

    # Case 2: inhibition
    if weight < 0:
        c = inhib
        arc = Arc((x1,y1), d, d, angle=180, theta1=angle1, theta2=angle2-2,
        lw=1.1, color=c, zorder=0)
        ax.add_artist(arc)

        # Inhibition head
        width = 0.06
        height = 0.015

        angle = angle2 - 180 - 2
        theta = angle * (np.pi/180)

        x = x1 - width/2 + d/2
        y = y1 - height/2

        x2 = x1 + np.cos(theta)*(x-x1) - np.sin(theta)*(y-y1)
        y2 = y1 + np.sin(theta)*(x-x1) + np.cos(theta)*(y-y1)

        head = Rectangle((x2,y2), width, height, angle=angle, fc=c, zorder=0)
        ax.add_artist(head)


# Main function
def plot_network(inter, width=1, height=1, vdict=None,
    tol=None, root=False, file=None, verb=False):
    """
    Plot a network.
    """
    n, n = inter.shape
    w, h = width/2.54, height/2.54 # Centimeters
    if vdict is None: vdict = {}

    # Compute layout
    v = list(range(n))
    e = list(zip(*inter.nonzero()))
    p = graph_layout(v, e, w, h, tol=tol, root=root, verb=verb)

    # Draw layout
    fig = plt.figure(figsize=(w,h), dpi=100, frameon=False)
    plt.axes([0,0,1,1])
    ax = fig.gca()
    ax.axis('off')
    ax.axis('equal')
    plt.xlim([0, w])
    plt.ylim([0, h])

    # Draw nodes
    I, J = inter.nonzero()
    for k in set(I) | set(J):
        name = '{}'.format(k)
        node(k, ax, p, name)

    # Draw links
    for k1, k2 in e:
        weight = inter[k1,k2]
        if k1 != k2:
            if (k2, k1) in e: link(k1, k2, ax, p, weight, bend=0.2)
            else: link(k1, k2, ax, p, weight, bend=0)
        else:
            v = vdict.get(k1)
            if v is None:
                b, c = 0, 0
                for k in range(n):
                    if (k != k1) and (((k,k1) in e) or ((k1, k) in e)):
                        b += p[k]
                        c += 1
                v = p[k1] - b/c
            else: v = np.array(v)
            d = np.sqrt(np.sum(v**2))
            if d > 0: v = v/d
            else: v = np.array([0,1])
            link_auto(k1, ax, p, weight, v)
    if file is None: file = 'network.pdf'
    fig.savefig(file, dpi=100, bbox_inches=0, frameon=False)
    plt.close()


# Tests
if __name__ == '__main__':
    np.random.seed(0)

    # Simple graph example
    # v = list(range(10))
    # e = [(1,2),(0,9),(0,5),(1,4),(3,8),(0,6),(5,7)]
    # e = [(0,1),(0,3),(0,5)]
    # e = [(0,3),(0,5)]
    # e = []

    import sys; sys.path.append('../../../Harissa')
    from harissa.generator import tree

    n = 10
    w = np.ones((n+1,n+1))
    # w[1] = 10
    # w[2] = 10
    basal, inter = tree(n, weight=w)
    np.random.seed(0)
    # basal, inter = cascade(n)
    inter[2,8] = -1
    inter[8,2] = 1
    inter[3,5] = -1
    inter[5,3] = -1
    # inter[10,10] = 1
    for i in range(1,n+1):
        inter[i,i] = 1

    # vdict = {1:(-1,-0), 2:(1,0.6), 3:(-1,0.6), 5:(-1,0.2)}
    plot_network(inter, width=10, height=10, vdict=None,
        tol=1e-3, root=True, verb=False)
