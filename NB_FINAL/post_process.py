# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:50 2020

@author: Juan Camilo V
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

# Set plotting defaults
gray = '#757575'
plt.rcParams['image.cmap'] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.color"] = gray
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.color"] = gray
plt.rcParams["ytick.color"] = gray
plt.rcParams["axes.labelcolor"] = gray
plt.rcParams["axes.edgecolor"] = gray
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


#%% Plotting routines
def fields_plot(elements, nodes, disp, E_nodes=None, S_nodes=None):
    """Plot contours for displacements, strains and stresses

    Parameters
    ----------
    nodes : ndarray (float)
        Array with number and nodes coordinates:
         `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    disp : ndarray (float)
        Array with the displacements.
    E_nodes : ndarray (float)
        Array with strain field in the nodes.
    S_nodes : ndarray (float)
        Array with stress field in the nodes.

    """
    # Check for structural elements in the mesh
    
    plot_node_field(disp, nodes, elements, title=[r"$u_x$", r"$u_y$"],
                    figtitle=["Horizontal displacement",
                              "Vertical displacement"])
    if E_nodes is not None:
        plot_node_field(E_nodes, nodes, elements,
                        title=[r"$\epsilon_{xx}$",
                               r"$\epsilon_{yy}$",
                               r"$\gamma_{xy}$",],
                        figtitle=["Strain epsilon-xx",
                                  "Strain epsilon-yy",
                                  "Strain gamma-xy"])
    if S_nodes is not None:
        plot_node_field(S_nodes, nodes, elements,
                        title=[r"$\sigma_{xx}$",
                               r"$\sigma_{yy}$",
                               r"$\tau_{xy}$",],
                        figtitle=["Stress sigma-xx",
                                  "Stress sigma-yy",
                                  "Stress tau-xy"])




def tri_plot(tri, field, title="", levels=12, savefigs=False,
             plt_type="contourf", filename="solution_plot.pdf"):
    """Plot contours over triangulation

    Parameters
    ----------
    tri : ndarray (float)
        Array with number and nodes coordinates:
        `number coordX coordY BCX BCY`
    field : ndarray (float)
        Array with data to be plotted for each node.
    title : string (optional)
        Title of the plot.
    levels : int (optional)
        Number of levels to be used in ``contourf``.
    savefigs : bool (optional)
        Allow to save the figure.
    plt_type : string (optional)
        Plot the field as one of the options: ``pcolor`` or
        ``contourf``
    filename : string (optional)
        Filename to save the figures.
    """
    if plt_type == "pcolor":
        disp_plot = plt.tripcolor
    elif plt_type == "contourf":
        disp_plot = plt.tricontourf
    disp_plot(tri, field, levels, shading="gouraud")
    plt.title(title)
    plt.colorbar(orientation='vertical')
    plt.axis("image")
    if savefigs:
        plt.savefig(filename)


def plot_node_field(field, nodes, elements, plt_type="contourf", levels=12,
                    savefigs=False, title=None, figtitle=None,
                    filename=None):
    """Plot the nodal displacement using a triangulation

    Parameters
    ----------
    field : ndarray (float)
          Array with the field to be plotted. The number of columns
          determine the number of plots.
    nodes : ndarray (float)
        Array with number and nodes coordinates
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each  element.
    plt_type : string (optional)
        Plot the field as one of the options: ``pcolor`` or
        ``contourf``.
    levels : int (optional)
        Number of levels to be used in ``contourf``.
    savefigs : bool (optional)
        Allow to save the figure.
    title : Tuple of strings (optional)
        Titles of the plots. If not provided the plots will not have
        a title.
    figtitle : Tuple of strings (optional)
        Titles of the plotting windows. If not provided the
        windows will not have a title.
    filename : Tuple of strings (optional)
        Filenames to save the figures. Only used when `savefigs=True`.
        If not provided the name of the figures would be "outputk.pdf",
        where `k` is the number of the column.
    """
    tri = mesh2tri(nodes, elements)
    if len(field.shape) == 1:
        nfields = 1
    else:
        _, nfields = field.shape
    if title is None:
        title = ["" for cont in range(nfields)]
    if figtitle is None:
        figs = plt.get_fignums()
        nfigs = len(figs)
        figtitle = [cont + 1 for cont in range(nfigs, nfigs + nfields)]
    if filename is None:
        filename = ["output{}.pdf".format(cont) for cont in range(nfields)]
    for cont in range(nfields):
        if nfields == 1:
            current_field = field
        else:
            current_field = field[:, cont]
        plt.figure(figtitle[cont])
        tri_plot(tri, current_field, title=title[cont], levels=levels,
                 plt_type=plt_type, savefigs=savefigs,
                 filename=filename[cont])
        if savefigs:
            plt.savefig(filename[cont])


def plot_truss(nodes, elements, mats, stresses=None, max_val=4,
               min_val=0.5, savefigs=False, title=None, figtitle=None,
               filename=None):
    """Plot a truss and encodes the stresses in a colormap

    Parameters
    ----------
    UC : (nnodes, 2) ndarray (float)
      Array with the displacements.
    nodes : ndarray (float)
        Array with number and nodes coordinates
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each  element.
    mats : ndarray (float)
        Array with material profiles.
    loads : ndarray (float)
        Array with loads.
    tol : float (optional)
        Minimum difference between cross-section areas of the members
        to be considered different.
    savefigs : bool (optional)
        Allow to save the figure.
    title : Tuple of strings (optional)
        Titles of the plots. If not provided the plots will not have
        a title.
    figtitle : Tuple of strings (optional)
        Titles of the plotting windows. If not provided the
        windows will not have a title.
    filename : Tuple of strings (optional)
        Filenames to save the figures. Only used when `savefigs=True`.
        If not provided the name of the figures would be "outputk.pdf",
        where `k` is the number of the column.

    """
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    if stresses is None:
        scaled_stress = np.ones_like(elements[:, 0])
    else:
        max_stress = max(-stresses.min(), stresses.max())
        scaled_stress = 0.5*(stresses + max_stress)/max_stress
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    plt.figure(figtitle)
    for elem in elements:
        ini, end = elem[3:]
        color = plt.cm.seismic(scaled_stress[elem[0]])
        plt.plot([nodes[ini, 1], nodes[end, 1]],
                 [nodes[ini, 2], nodes[end, 2]],
                 color=color, lw=widths[elem[2]])

    if title is None:
        title = ''
    if figtitle is None:
        figtitle = ""
    if filename is None:
        filename = "output.pdf"
    plt.title(title)
    plt.axis("image")
    if savefigs:
        plt.savefig(filename)


#%% Auxiliar functions for plotting
def mesh2tri(nodes, elements):
    """Generate a  matplotlib.tri.Triangulation object from the mesh

    Parameters
    ----------
    nodes : ndarray (float)
      Array with number and nodes coordinates:
        `number coordX coordY BCX BCY`
    elements : ndarray (int)
      Array with the node number for the nodes that correspond to each
      element.

    Returns
    -------
    tri : Triangulation
        An unstructured triangular grid consisting of npoints points
        and ntri triangles.

    """
    coord_x = nodes[:, 1]
    coord_y = nodes[:, 2]
    triangs = []
    if len(elements.shape) == 1:
        triangs.append(elements[[1, 5, 8]])
        triangs.append(elements[[5, 2, 6]])
        triangs.append(elements[[5, 6, 8]])
        triangs.append(elements[[8, 7, 4]])
        triangs.append(elements[[6, 3, 7]])
        triangs.append(elements[[8, 6, 7]])
    else:  
        for elem in elements:
        #     if elem[1] == 1:
            triangs.append(elem[[1, 5, 8]])
            triangs.append(elem[[5, 2, 6]])
            triangs.append(elem[[5, 6, 8]])
            triangs.append(elem[[8, 7, 4]])
            triangs.append(elem[[6, 3, 7]])
            triangs.append(elem[[8, 6, 7]])
        #     if elem[1] == 2:
        #         triangs.append(elem[[3, 6, 8]])
        #         triangs.append(elem[[6, 7, 8]])
        #         triangs.append(elem[[6, 4, 7]])
        #         triangs.append(elem[[7, 5, 8]])
            # if elem[1] == 3:
            #     triangs.append(elem[3:])

    tri = Triangulation(coord_x, coord_y, np.array(triangs))
    print(tri)
    return tri


#%% Auxiliar variables computation
def complete_disp(IBC, nodes, UG):
    """
    Fill the displacement vectors with imposed and computed values.

    IBC : ndarray (int)
        IBC (Indicator of Boundary Conditions) indicates if the
        nodes has any type of boundary conditions applied to it.
    UG : ndarray (float)
        Array with the computed displacements.
    nodes : ndarray (float)
        Array with number and nodes coordinates

    Returns
    -------
    UC : (nnodes, 2) ndarray (float)
      Array with the displacements.

    """
    nnodes = nodes.shape[0]
    UC = np.zeros([nnodes, 2], dtype=np.float)
    for row in range(nnodes):
        for col in range(2):
            cons = IBC[row, col]
            if cons == -1:
                UC[row, col] = 0.0
            else:
                UC[row, col] = UG[cons]

    return UC





