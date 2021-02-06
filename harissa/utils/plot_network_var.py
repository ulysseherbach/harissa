"""
Plotting networks - Simple variant with undirected and colored edges
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.patches import ArrowStyle

activ = '#96D255'
inhib = '#E97356'

def circ_layout(n, width, height):
    """Circular graph layout"""
    p = np.zeros((n,2))
    a = np.pi/2 + 2 * np.pi * np.linspace(0, 1, n+1)[:n] - 0.000
    p[:,0] = (0.5 * (1 - 2/6) * np.cos(a) + 1/2) * width
    p[:,1] = (0.5 * (1 - 2/6) * np.sin(a) + 1/2) * height
    return p

# Plotting functions
def node(k, ax, p, name, scale=1, color=None, fontsize=None):
    if color is None: color = 'gray'
    if fontsize is None: fontsize = 8*scale
    r = 0.1 * scale
    x, y = p[k,0] * scale, p[k,1] * scale
    circle1 = plt.Circle((x, y), radius=r, fill=False,
        color='lightgray', lw=1*scale, zorder=3, clip_on=False)
    ax.add_artist(circle1)
    # ax.text(x, y - 0.007*scale, name, color=color,
    #     fontsize=fontsize, clip_on=False, zorder=20,
    #     horizontalalignment='center', verticalalignment='center')
    ax.text(x, y, name, color=color,
        fontsize=fontsize, clip_on=False, zorder=20,
        horizontalalignment='center', verticalalignment='center')

def link(k1, k2, ax, p, weight, bend=0, scale=1, undir=False, color=None):
    # Node coordinates
    x1, y1 = p[k1,0]*scale, p[k1,1]*scale
    x2, y2 = p[k2,0]*scale, p[k2,1]*scale

    # Case 1: activation
    fc = activ if color is None else color
    if weight > 0:
        if undir:
            style = ArrowStyle('Simple', tail_width=1.1*scale,
                head_width=0, head_length=1e-9*scale)
        else:
            style = ArrowStyle('Simple', tail_width=1.1*scale,
                head_width=3.5*scale, head_length=5*scale)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=9*scale, shrinkB=9*scale, fc=fc, lw=0, zorder=10,
        connectionstyle='arc3,rad={}'.format(bend), clip_on=False)
        ax.add_artist(arrow)

    # Case 2: inhibition
    fc = inhib if color is None else color
    if weight < 0:
        style = ArrowStyle('Simple', tail_width=1.1*scale,
            head_width=0, head_length=1e-9*scale)
        arrow = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle=style,
        shrinkA=9*scale, shrinkB=9*scale, fc=fc, lw=0, zorder=10,
        connectionstyle='arc3,rad={}'.format(bend), clip_on=False)
        ax.add_artist(arrow)

        r = 0.125
        u = p[k2] - p[k1]
        u0 = u/np.sqrt(np.sum(u**2))
        x0 = (p[k1,0] + r*u0[0])*scale
        y0 = (p[k1,1] + r*u0[1])*scale
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

        head = Rectangle((x3,y3), h_width, h_height,
            angle=angle, fc=fc, zorder=0, clip_on=False)
        if not undir: ax.add_artist(head)

# Main function
def plot_network(inter, width=1, height=1, scale=1, names=None, vdict=None,
    tol=None, root=False, axes=None, layout=None, nodes=None, n0=True,
    file=None, verb=False, vcolor=None, ecolor=None, cmap=None):
    """
    Plot a network.
    """
    G, G = inter.shape
    w, h = width/2.54, height/2.54 # Centimeters
    if names is None: names = ['{}'.format(k) for k in range(G)]
    if vdict is None: vdict = {}

    # Compute layout
    v = list(range(G))
    e = list(zip(*inter.nonzero()))
    if layout is None: p = circ_layout(G, width, height)
    else: p = layout

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
    pos = ax.get_position()
    plt.xlim([0, w*pos.width])
    plt.ylim([0, h*pos.height])
    scale = scale * np.min([pos.width,pos.height])

    # Draw nodes
    # I, J = set(I), set(J)
    # if not n0: I, J = I-{0}, J-{0}
    # for k in I | J:
    #     node(k, ax, p, names[k], scale)
    for k in range(G):
        if vcolor is not None: color = vcolor[k]
        else: color = None
        node(k, ax, p, names[k], scale, color=color)

    # Draw links
    for k1, k2 in e:
        weight = inter[k1,k2]
        if ecolor is not None:
            if cmap is None: color = ecolor
            else: color = cmap[ecolor[k1,k2]]
        else: color = None
        if k1 != k2:
            if (k2, k1) in e:
                if k1 < k2:
                    link(k1, k2, ax, p, weight, 0, scale, True, color=color)
            else: link(k1, k2, ax, p, weight, 0, scale, color=color)
        else:
            v = vdict.get(k1)
            if v is None:
                b, c = 0, 0
                for k in range(G):
                    if (k != k1) and (((k,k1) in e) or ((k1, k) in e)):
                        b += p[k]
                        c += 1
                if c == 0:
                    b, c = 0, 0
                    for k in I | J:
                        if (k != k1):
                            b += p[k]
                            c += 1
                v = p[k1] - b/c
            else: v = np.array(v)
            d = np.sqrt(np.sum(v**2))
            if d > 0: v = v/d
            else: v = np.array([0,1])
            # link_auto(k1, ax, p, weight, v, scale)
    if file is None: file = 'network.pdf'
    if axes is None: fig.savefig(file, dpi=100, bbox_inches=0, frameon=False)
