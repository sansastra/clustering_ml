# -*- coding: utf-8 -*-
# @Time    : 06.04.21 10:42
# @Author  : sing_sd

import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from train import train_model
from test import test_data
from net_cluster import Net
from sklearn.model_selection import train_test_split

def main():
    features = ['x', 'y', 'cog', 'sog'] # extra features in data= 'mmsi', 'cluster'
    dim = len(features)
    timesteps = 10  # number of sequential features per sample
    # load data
    data  = pd.read_csv("dbscan_clustered_data.csv") # "dbscan_clustered_data-full.csv has one class with very high data, acc-93%
    num_classes = len(data.cluster.unique())
    # data_normal, Y_data = np.array(data.iloc[:,0:4]), np.array(data.iloc[:,5])
    data_normal, Y_data = load_data(np.array(data), num_classes, features, timesteps)
    np.random.seed(1234)
    x_train, x_test, y_train, y_test = train_test_split(data_normal, Y_data, test_size=0.10)
    train = True
    if train:
        num_epochs = 50
        model = Net(dropout=False, input_size=timesteps*dim, output_size=num_classes)
        criterion = nn.BCEWithLogitsLoss() #CrossEntropyLoss()  # nn.MSELoss()  #
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        model, metrics = train_model(model, x_train, x_test, y_train, y_test, num_classes, criterion,
                                     optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs)

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(state, "./results/model.pt")
        print("Saved: ./results/model.pt")

    ######## test ###############################################################
    else:
        model = Net(dropout=False, input_size=timesteps*dim, output_size=num_classes)
        optimizer = optim.Adam(model.parameters())
        checkpoint = torch.load("./results/model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.eval()
        test_data(model, x_test, y_test)


def load_data(data, num_classes, features, timesteps):
    dim = len(features)
    mmsi_idx = 4
    output_idx = 5
    clusters = np.unique(data[:, output_idx])
    for cluster in clusters:
        cluster_idx = data[:, output_idx] == cluster
        print("data for cluster nr ", cluster, " is=", np.sum(cluster_idx))

    mmsis = np.unique(data[:, mmsi_idx])
    x_data = np.zeros(shape=(2*len(data), timesteps*dim))
    y_data = np.zeros(shape=(2*len(data), num_classes))
    index= 0
    for mmsi in mmsis:
        data_vessel_idx = data[:, mmsi_idx] == mmsi
        nr_data = np.sum(data_vessel_idx)
        if nr_data < timesteps:
            continue
        x_data[index: index + nr_data, 0:dim] = data[data_vessel_idx, 0:dim]

        for clm_nr in range(1, timesteps):
            x_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
                x_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]


        target = data[data_vessel_idx, output_idx]
        for i in range(nr_data-timesteps+1):
            y_data[i+index, int(target[i])] = 1.0

        index += nr_data - timesteps + 1

        # reverse dataset
        x_data[index: index + nr_data, 0:dim] =  np.flipud(data[data_vessel_idx, 0:dim])
        for clm_nr in range(1, timesteps):
            x_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
                x_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

        for i in range(nr_data-timesteps+1):
            y_data[i+index, int(target[-i])] = 1.0
        index += nr_data - timesteps + 1
        # x_data = np.delete(x_data,range(index, len(data)), axis=0)
    idx_delete = np.where((np.sum(y_data, axis=1) < 0.5))[0]
    if len(np.where((np.sum(y_data, axis=0) < 0.5))[0]) > 0:
        print("there is no data for some classes")
    x_data = np.delete(x_data, idx_delete, axis=0)
    y_data = np.delete(y_data, idx_delete, axis=0)
    return x_data, y_data

if __name__ == "__main__":
    main()
