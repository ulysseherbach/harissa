"""
Plotting networks
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc, Polygon
from matplotlib.patches import ArrowStyle

# Edge colors
activ = plt.get_cmap('tab10')(2)
inhib = plt.get_cmap('tab10')(3)

def build_pos(inter, method=None):
    """
    Compute node layout given an interaction matrix.
    """
    G = inter.shape[0]
    # Define graph
    graph = nx.Graph()
    for i in range(G):
        graph.add_node(i)
        for j in range(0,G):
            graph.add_node(j)
            if np.abs(inter[i,j] != 0):
                graph.add_edge(i, j)
    # Compute graph layout
    if method is None:
        p0 = np.random.normal(size=(G,2))
        p = nx.kamada_kawai_layout(graph, pos=p0)
    if method == 'graphviz':
        p = nx.nx_agraph.graphviz_layout(graph, prog='neato')
    # Return node positions
    pos = np.zeros((G,2))
    for k in range(0,G):
        pos[k,:] = p[k]
    scale = 1 if method is None else 2/(pos.max()-pos.min())
    center = pos.mean(axis=0)
    return scale * (pos - center)

#### Plotting functions ####

def node(k, ax, pos, name, scale=1, color=None, fontsize=None, nodesize=1):
    if color is None:
        color = 'gray'
    if fontsize is None:
        fontsize = 8*scale
    x, y = pos[k,0] * scale, pos[k,1] * scale
    r = nodesize * 0.1 * scale
    # Node shape
    circle1 = plt.Circle((x, y), radius=r, fill=True, facecolor='white',
        edgecolor='lightgray', lw=1*scale, zorder=3, clip_on=False)
    ax.add_artist(circle1)
    # Node label
    ax.text(x, y - 0.007*scale, name, color=color, fontsize=fontsize,
        zorder=4, horizontalalignment='center', verticalalignment='center')

def link(k1, k2, ax, pos, weight, bend=0, scale=1, nodesize=1, alpha=None):
    # Node coordinates
    x1, y1 = pos[k1,0]*scale, pos[k1,1]*scale
    x2, y2 = pos[k2,0]*scale, pos[k2,1]*scale
    shrink = 9 * scale * nodesize
    if alpha is None: alpha = 1

    # Case 1: activation
    if weight > 0:
        style = ArrowStyle('Simple', tail_width=1.1*scale,
            head_width=3.5*scale, head_length=5*scale)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=shrink, shrinkB=shrink, fc=activ, lw=0, zorder=0,
        connectionstyle='arc3,rad={}'.format(bend),
        clip_on=False, alpha=alpha)
        ax.add_artist(arrow)

    # Case 2: inhibition
    if weight < 0:
        style = ArrowStyle('Simple', tail_width=1.1*scale,
            head_width=0, head_length=1e-9*scale)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=shrink, shrinkB=shrink, fc=inhib, lw=0, zorder=0,
        connectionstyle='arc3,rad={}'.format(bend),
        clip_on=False, alpha=alpha)
        ax.add_artist(arrow)

        r = 0.125*nodesize
        u = pos[k2] - pos[k1]
        u0 = u/np.sqrt(np.sum(u**2))
        x0 = (pos[k1,0] + r*u0[0])*scale
        y0 = (pos[k1,1] + r*u0[1])*scale
        dx = (u[0] - 2*r*u0[0])*scale
        dy = (u[1] - 2*r*u0[1])*scale

        h_width = 0.07*scale
        h_height = 0.015*scale
        x1, y1 = x0 + dx, y0 + dy
        d = np.sqrt(dx**2 + dy**2)
        u = np.array([dx, dy])/d
        v = np.array([dy,-dx])/d

        x = x1 + 0.2*h_height*u[0] + 0.5*h_width*v[0]
        y = y1 + 0.2*h_height*u[1] + 0.5*h_width*v[1]

        if v[0] > 0: angle = (180/np.pi)*np.arctan(v[1]/v[0]) - 180
        elif v[0] < 0: angle = (180/np.pi)*np.arctan(v[1]/v[0])
        else: angle = np.sign(u[0]) * 90

        angle += np.tanh(bend) * 90
        theta = np.tanh(bend) * (np.pi/2)

        x3 = x2 + np.cos(theta)*(x-x2) - np.sin(theta)*(y-y2)
        y3 = y2 + np.sin(theta)*(x-x2) + np.cos(theta)*(y-y2)

        head = Rectangle((x3,y3), h_width, h_height, clip_on=False,
            angle=angle, fc=inhib, zorder=0, alpha=alpha)
        ax.add_artist(head)

def link_auto(k, ax, pos, weight, v=None, scale=1, nodesize=1, alpha=None):
    # Node coordinates
    x0, y0 = pos[k,0]*scale, pos[k,1]*scale

    # Orientation
    if v is None:
        v = np.array([1,0])
        v /= np.sqrt(np.sum(v**2))

    if v[0] > 0: angle0 = np.arcsin(v[1]) * (180/np.pi)
    else: angle0 = 180 - np.arcsin(v[1]) * (180/np.pi)

    angle1 = 58 + angle0
    angle2 = 300 + angle0
    theta = angle0 * (np.pi/180)

    x, y = x0 + 0.145*scale, y0
    x1 = x0 + np.cos(theta)*(x-x0) - np.sin(theta)*(y-y0)
    y1 = y0 + np.sin(theta)*(x-x0) + np.cos(theta)*(y-y0)

    # Loop diameter
    d = 0.15*scale

    # Case 1: activation
    if weight > 0:
        c = activ
        arc = Arc((x1,y1), d, d, angle=180, theta1=angle1, theta2=angle2-30,
        lw=1.1*scale, color=c, zorder=0, clip_on=False)
        ax.add_artist(arc)

        # Activation head
        head_width = 0.045*scale
        head_length = 0.065*scale

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

        head = Polygon(triangle, fc=activ, zorder=0, clip_on=False)
        ax.add_artist(head)

    # Case 2: inhibition
    if weight < 0:
        c = inhib
        arc = Arc((x1,y1), d, d, angle=180, theta1=angle1, theta2=angle2-2,
        lw=1.1*scale, color=c, zorder=0, clip_on=False)
        ax.add_artist(arc)

        # Inhibition head
        width = 0.06*scale
        height = 0.015*scale

        angle = angle2 - 180 - 2
        theta = angle * (np.pi/180)

        x = x1 - width/2 + d/2
        y = y1 - height/2

        x2 = x1 + np.cos(theta)*(x-x1) - np.sin(theta)*(y-y1)
        y2 = y1 + np.sin(theta)*(x-x1) + np.cos(theta)*(y-y1)

        head = Rectangle((x2,y2), width, height, angle=angle, fc=c,
            zorder=0, clip_on=False)
        ax.add_artist(head)

#### Main function ####

def plot_network(inter, pos, width=1, height=1, scale=1, names=None,
    vdict=None, tol=None, root=False, axes=None, nodes=None, n0=True,
    file=None, verb=False, fontsize=None, vcolor=None, nodesize=1,
    bend=0.14, bend_all=False, alpha=None):
    """
    Plot a network.
    """
    G, G = inter.shape
    w, h = width/2.54, height/2.54 # Centimeters
    if names is None: names = [''] + ['{}'.format(k) for k in range(1,G)]
    if vcolor is None: vcolor = ['#5C5C5C' for k in range(G)]
    if vdict is None: vdict = {}
    # Compute layout
    v = list(range(G))
    e = list(zip(*inter.nonzero()))
    # Draw layout
    if axes is None:
        fig = plt.figure(figsize=(w,h), dpi=100, frameon=False)
        plt.axes([0,0,1,1])
        ax = fig.gca()
        I, J = inter.nonzero()
        ax.axis('off')
    else:
        ax = axes
        fig = plt.gcf()
        size = fig.get_size_inches()
        w, h = size[0], size[1]
        if nodes is None: I, J = inter.nonzero()
        else: I, J = nodes, nodes
    ax.axis('equal')
    ax.axis('off')
    ax_pos = ax.get_position()
    plt.xlim([-w*ax_pos.width/2, w*ax_pos.width/2])
    plt.ylim([-h*ax_pos.height/2, h*ax_pos.height/2])
    scale = scale * np.min([ax_pos.width,ax_pos.height])
    # Draw nodes
    for k in range(G):
        node(k, ax, pos, names[k], scale, fontsize=fontsize, color=vcolor[k],
            nodesize=nodesize)
    # Draw links
    for k1, k2 in e:
        weight = inter[k1,k2]
        if k1 != k2:
            if (k2, k1) in e:
                link(k1, k2, ax, pos, weight, bend, scale, nodesize=nodesize,
                    alpha=alpha)
            else: link(k1, k2, ax, pos, weight, bend*bend_all, scale,
                nodesize=nodesize, alpha=alpha)
        else:
            v = vdict.get(k1)
            if v is None:
                b, c = 0, 0
                for k in range(G):
                    if (k != k1) and (((k,k1) in e) or ((k1, k) in e)):
                        b += pos[k]
                        c += 1
                if c == 0:
                    b, c = 0, 0
                    for k in I | J:
                        if (k != k1):
                            b += pos[k]
                            c += 1
                v = pos[k1] - b/c
            else: v = np.array(v)
            d = np.sqrt(np.sum(v**2))
            if d > 0: v = v/d
            else: v = np.array([0,1])
            link_auto(k1, ax, pos, weight, v, scale, nodesize=nodesize,
                alpha=alpha)
    if file is None: file = 'network.pdf'
    if axes is None: fig.savefig(file, bbox_inches='tight')
