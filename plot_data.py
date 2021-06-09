# -*- coding: utf-8 -*-
# @Time    : 07.04.21 11:47
# @Author  : sing_sd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import PythonCode.Jrnl.COStransforms as ct

ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}
plt.rcParams.update({'font.size': 12})

def main():
    data = genfromtxt('ais_data_1min_graph.csv', delimiter=',')
    plot_graph(data)


def plot_graph(data):
    colour_array = ["r", "g", "b", "y", "c", "m", "#9475FC", "#a23456"] # an extra
    clusters = np.unique(data[:, -2])
    i = 0
    if len(colour_array) < len(clusters):
        print("increase the clorors in color_array")
        exit(0)

    for cluster in clusters:
        cluster_idx = data[:,-2] == cluster

        plt.plot(data[cluster_idx, 0], data[cluster_idx, 1], ".",color=colour_array[i], markersize=2)
        i += 1

    nodes = get_mulist()
    nodes = ct.ENUtoWGS84(np.array(nodes).T, tREF=ref)
    nodes = np.delete(nodes,  np.s_[2], axis=0)
    nodes = nodes.T
    edges = get_assignment_edges()
    for ee, e in enumerate(edges):
        plt.plot([nodes[e[0]][0], nodes[e[1]][0]], [nodes[e[0]][1], nodes[e[1]][1]], color="k", marker=".", markersize=10)
        plt.plot([nodes[e[0]][0], nodes[e[1]][0]], [nodes[e[0]][1], nodes[e[1]][1]], "-", color= "k", linewidth=4) #colour_array[ee]
        plt.pause(0.0001)

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.xlim(11.3, 12.4)
    plt.ylim(54.1, 54.6)
    plt.show()

def get_mulist():
    return [[-35000, 20000, 0],
            [-14000, 8000, 0],
            [10500, 10000, 0],
            [20000, 25000, 0],
            [5400, -15500, 0],
            [-4500, 24000, 0],
            [-35000, -20000, 0]]
            #[4000, 9400, 0]]


def get_assignment_edges():
    return [[0, 1], [1, 2], [2, 3], [1, 4], [2, 4], [4, 5], [2, 6]]

if __name__ == "__main__":
    main()
