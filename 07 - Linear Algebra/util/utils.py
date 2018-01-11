import numpy as np

import matplotlib.pyplot as plt

from IPython.core.display import HTML

import cmath
import time

def make_mesh(mesh_props):
    num_dimensions = 2

    mins = (mesh_props['x_min'],mesh_props['y_min'])
    maxs = (mesh_props['x_max'],mesh_props['y_max'])
    delta = mesh_props['delta']

    for idx in range(num_dimensions):
        assert mins[idx] < maxs[idx], "min can't be bigger than max!"

    ranges = [np.arange(mins[idx],maxs[idx]+delta,delta) for idx in range(num_dimensions)]

    xs,ys = np.meshgrid(*ranges)

    return xs,ys

def plot_mesh(f,ax,xs,ys,colors):
    ax.set_aspect('equal')
    h = plt.scatter(xs.flatten(),ys.flatten(),
                alpha=0.7,edgecolor='none',
                s=36,linewidth=2,
                zorder=6,
               c=colors,cmap='hot')

    return h

def plot_vector(v,color,label=None):
    return plt.arrow(0,0,v[0],v[1],zorder=5,
              linewidth=6,color=color,head_width=0.1,label=label)

def plot_interesting_vectors(T,columns=False,eigs=True):
    arrows = []
    labels = []
    if columns:
        arrows += [plot_vector(column,'hotpink')
                       for column in T.T]
        arrows = [arrows[0]]
        labels += ["a basis vector lands here"]
    if eigs:
        eigenvalues,eigenvectors = np.linalg.eig(T)
        eigen_list = [(eigenvalue,eigenvector) for eigenvalue,eigenvector
                                    in zip(eigenvalues,eigenvectors.T)
                                  if eigenvalue != 0
                                      and not(np.iscomplex(eigenvalue))
                                      ]
        if eigen_list:
            eigen_arrows = [plot_vector(np.real(element[1]),'#53fca1',label='special vectors')
                                 for element
                                     in eigen_list
                                     ]
            eigen_arrows = [eigen_arrows[0]]
            labels += ["this is a special (aka eigen) vector"]
            arrows += eigen_arrows
        else:
            print("eigenvalues are all nonreal or 0")
    plt.legend(arrows,labels,loc=[0,0.5],
               bbox_to_anchor=(0,1.01),
               ncol=1,prop={'weight':'bold','size':'xx-small'})
    return

def compute_trajectories(T,scatter):

    starting_positions = scatter.get_offsets()
    ending_positions = np.dot(T,starting_positions.T).T
    delta_positions = ending_positions-starting_positions

    return starting_positions,ending_positions,delta_positions

def set_axes_lims(mn,mx,ax=None):
    if ax == None:
        ax = plt.gca()

    ax.set_ylim([mn,mx])
    ax.set_xlim([mn,mx])

    return

def calculate_axis_bounds(starting_positions,ending_positions,buffer_factor=1.1):
    #axis bounds to include starting and ending positions of each point

    mn = buffer_factor*min(np.min(starting_positions),np.min(ending_positions))
    mx = buffer_factor*max(np.max(starting_positions),np.max(ending_positions))

    if mn == 0:
        mn -= 0.1
    if mx == 0:
        mx += 0.1

    return mn,mx

def draw_coordinate_axes(mn,mx,ax=None):
    if ax == None:
        ax = plt.gca()

    plt.hlines(0,mn,mx,zorder=4,linewidth=4,color='grey')
    plt.vlines(0,mn,mx,zorder=4,linewidth=4,color='grey')

    return

def make_rotation(theta):
    rotation_matrix = [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    return np.asarray(rotation_matrix)

unit_square_mesh = {'delta':0.1,
                    'x_min' :0,
                    'x_max' : 1,
                    'y_min' : 0,
                    'y_max' : 1,}

foursquare_mesh = {'delta':0.1,
                    'x_min' :-0.5,
                    'x_max' : 0.5,
                    'y_min' : -0.5,
                    'y_max' : 0.5,}

def setup_plot(T,mesh_properties=unit_square_mesh,square_axes=False,
             plot_columns=False,plot_eigenvectors=False):
    """
    Setup the plot and axes for animating a linear transformation T.

    If asked, plot the columns (aka the images of the basis vectors)
        and the eigenvectors (but only if they're real and non-zero).

    Parameters
    ----------
    T        : 2x2 matrix representing a linear transformation
    mesh_properties : dictionary that defines properties of meshgrid of points
                        that will be plotted and transformed.
                        needs to have the following five properties:
                        'delta' - mesh spacing
                        '{x,y}{Min,Max}' - minium/maximum value on x/y axis
    square_axes : if False, size the axes so that they contain starting
                    and ending location of each point in grid.
                 if True, size the axes so that they are square and contain
                    starting and ending location of each point in mesh.
    plot_columns : if True, plot the columns of the transformation so that we can see
                        where the basis vectors end up
    plot_eigenvectors: if true, plot the eigenvectors of the transformation

    Returns
    -------
    returns are meant to be consumed by animate_transformation

    scatter   : a PathCollection with all of the points in the meshgrid
    f         : matplotlib figure containing axes
    ax        : matplotlib axes containing scatter
    """
    T = np.asarray(T)

    xs,ys = make_mesh(mesh_properties)
    colors = np.linspace(0,1,num=xs.shape[0]*xs.shape[1])

    f = plt.figure(figsize=(6,6))
    ax = plt.gca()

    scatter = plot_mesh(f,ax,xs,ys,colors)

    plot_vectors = [plot_columns,plot_eigenvectors]

    not_zeros = not(np.all(T == np.zeros(2)))

    if (any(plot_vectors) & not_zeros):
        plot_interesting_vectors(T,*plot_vectors)

    start,end,delta = compute_trajectories(T,scatter)

    mn,mx = calculate_axis_bounds(start,end)

    if square_axes:
        lim = max(abs(mn),abs(mx))
        mn = -lim; mx = lim

    draw_coordinate_axes(mn,mx)
    set_axes_lims(mn,mx,ax=ax)

    f.canvas.draw()
    time.sleep(1)

    return scatter,f,ax


def animate_transformation(T,scatter,figure,
                          mesh_properties=unit_square_mesh,
                          delta_t=0.05,delay=0.,):
    """
    Animate a linear transformation T acting on points scatter in figure.

    If asked, plot the columns (aka the images of the basis vectors)
        and the eigenvectors (but only if they're real and non-zero).

    Parameters
    ----------
    T        : 2x2 matrix representing a linear transformation
    mesh_properties : dictionary that defines properties of meshgrid of points
                        that will be plotted and transformed.
                        needs to have the following five properties:
                        'delta' - mesh spacing
                        '{x,y}{_min,_max}' - minium/maximum value on x/y axis
    delta_t   : size of simulated timestep -- transformation complete at t = 1
    delay    : if non-zero, python will pause between frames of the animation for
                that many seconds.

    Returns
    -------
    returns are meant to be consumed by animate_transformation

    scatter   : a PathCollection with all of the points in the meshgrid
    f         : matplotlib figure containing axes
    ax        : matplotlib axes containing scatter
    """


    T = np.asarray(T)

    start,_,delta = compute_trajectories(T,scatter)

    I = np.eye(2)

    ts = np.arange(0,1+delta_t,delta_t)

    not_zeros = not(np.all(T == np.zeros(2)))

    if ((T[0,0] == T[1,1]) & (T[0,1] == -1*T[1,0])) & \
            not_zeros :

        z = complex(T[0,0],T[1,0])
        dz = z**(1/len(ts))

        dT = [[dz.real,-dz.imag],[dz.imag,dz.real]]
        for idx,t in enumerate(ts):
            current_positions = scatter.get_offsets()
            dT_toN = np.linalg.matrix_power(dT,idx+1)
            scatter.set_offsets(np.dot(dT_toN,start.T).T)
            figure.canvas.draw()
            if delay:
                time.sleep(delay)

    else:
        for idx,t in enumerate(ts):
            current_positions = scatter.get_offsets()
            scatter.set_offsets(t*(np.dot(T,start.T).T) +
                                   (1-t)*np.dot(I,start.T).T)

            figure.canvas.draw()
            if delay:
                time.sleep(delay)
