"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
import yaml


def anomaly_generation(anomaly_percent, data, user_data, rate_data, user_n, rate_n, time, rate, labels, cfg):

    np.random.seed(1)

    # knot2_user = int(data.shape[0] * (cfg['train_ratio'] + cfg['val_ratio']))
    # knot2_user = int(data.shape[0] * (cfg['train_ratio']))
    user_test_idx = range(0, data.shape[0])

    # select the other edges as the testing set
    test = data[user_test_idx]
    test_rate = np.expand_dims(rate[user_test_idx], axis=1)
    test_time = np.expand_dims(time[user_test_idx], axis=1)
    test_labels = np.expand_dims(labels[user_test_idx], axis=1)
    test_data = np.concatenate((test, test_time, test_rate, test_labels), axis=1)

    # generate fake edges that are not exist in the whole graph, treat them as
    # user edges anomalies
    user_idx_1 = np.expand_dims(np.transpose(np.random.choice(user_n, int(test.shape[0] * anomaly_percent))), axis=1)  # from [0, n-1]
    user_idx_2 = np.expand_dims(np.transpose(np.random.choice(user_n, int(test.shape[0] * anomaly_percent))), axis=1)  # m * 1
    user_generate_edges = np.concatenate((user_idx_1, user_idx_2), axis=1)
    user_fake_edges = processEdges(user_generate_edges, user_data)
    user_flag = np.expand_dims(np.array([1]).repeat(user_fake_edges.shape[0]), 1)

    user_time = np.expand_dims(np.transpose(np.random.choice(np.unique(time[user_test_idx]), user_fake_edges.shape[0])), axis=1)
    user_rateing = np.expand_dims(np.transpose(np.random.choice(np.array([0, 1]), user_fake_edges.shape[0])), axis=1)
    user_labels = np.expand_dims(np.array([0]).repeat(user_fake_edges.shape[0]), axis=1)
    user_fake_edges = np.concatenate((user_fake_edges, user_flag, user_time, user_rateing, user_labels), axis=1)

    #rate edges anomalies
    rate_idx_1 = np.expand_dims(np.transpose(np.random.choice(user_n, int(test.shape[0] * anomaly_percent))), axis=1)  # from [0, n-1]
    rate_idx_2 = np.expand_dims(np.transpose(np.random.choice(rate_n, int(test.shape[0] * anomaly_percent))), axis=1)  # m * 1
    rate_generate_edges = np.concatenate((rate_idx_1, rate_idx_2), axis=1)
    rate_fake_edges = processEdges(rate_generate_edges, rate_data)
    rate_flag = np.expand_dims(np.array([2]).repeat(rate_fake_edges.shape[0]), 1)

    rate_time = np.expand_dims(np.transpose(np.random.choice(np.unique(time[user_test_idx]), rate_fake_edges.shape[0])), axis=1)
    rate_rateing = np.expand_dims(np.array([1]).repeat(rate_fake_edges.shape[0]), axis=1)
    rate_labels = np.expand_dims(np.array([0]).repeat(rate_fake_edges.shape[0]), axis=1)
    rate_fake_edges = np.concatenate((rate_fake_edges, rate_flag, rate_time, rate_rateing, rate_labels), axis=1)

    synthetic_test = np.concatenate((test_data, user_fake_edges, rate_fake_edges), axis=0)
    synthetic_test = np.stack(sorted(synthetic_test, key=lambda line: line[3]))

    # synthetic_test = np.concatenate((test_data, fake_edges), axis=0)
    # synthetic_test = np.stack(sorted(synthetic_test, key=lambda line: line[3]))
    # anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    # anomalies = fake_edges[0:anomaly_num, :]

    # idx_test = np.ones([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # # randsample: sample without replacement
    # # it's different from datasample!

    # anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    # # anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200

    # idx_test[anomaly_pos] = 0
    # synthetic_test = np.concatenate(
    #     (np.zeros([np.size(idx_test, 0), 5], dtype=np.int32), idx_test), axis=1)

    # idx_anomalies = np.nonzero(idx_test.squeeze() == 0) #abnor: 0
    # idx_normal = np.nonzero(idx_test.squeeze() == 1)    #normal: 1

    # synthetic_test[idx_anomalies, 0:5] = anomalies
    # synthetic_test[idx_normal, 0:5] = test_data
    return synthetic_test


def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    # tmp = fake_edges[idx_fake]
    # tmp[:, [0, 1]] = tmp[:, [1, 0]]

    # fake_edges[idx_fake] = tmp

    idx_remove_dups = np.concatenate((np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)[0], 
                                    np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)[0]))

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    print('processEdges, for....')
    for i in a:
        if i not in b:
            c.append(i)
    print('Finally done....')
    fake_edges = np.array(c)
    fake_edges = np.unique(fake_edges, axis=0)
    return fake_edges


def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))
    n = np.max(data) + 1
    # n = max(max(user, item) for user, item in data)  # Get size of matrix

    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user][item] = 1  # Convert to 0-based index.
        matrix[item][user] = 1  # Convert to 0-based index.
    return matrix


def main_all(all_data, cfg):

    # all_data: userID(MyID), userID(otherID), flag(1, 2),  time, rating(1, 0), labels(1, 0),
    all_data = all_data.cpu()
    user_ori = np.concatenate((all_data[all_data[:, 2] == 1][:, [0]], all_data[all_data[:, 2] == 1][:, [1]], all_data[all_data[:, 2] == 2][:, [0]]))
    rate_ori = all_data[all_data[:, 2] == 2][:, [1]]
    user_edges = all_data[all_data[:, 2] == 1][:, [0, 1]]
    rate_edges = all_data[all_data[:, 2] == 2][:, [0, 1]]
    time = all_data[:, 3]
    rate = all_data[:, 4]
    labels = all_data[:, 5]
    user_vertices = np.unique(user_ori)
    rate_vertices = np.unique(rate_ori)
    # user_m = len(user_edges)
    user_n = len(user_vertices)
    # rate_m = len(rate_edges)
    rate_n = len(rate_vertices)
    edges = all_data[:, [0, 1, 2]]
    synthetic_test = anomaly_generation(
        0.05, edges, user_edges, rate_edges, user_n, rate_n, time, rate, labels, cfg)

    np.savetxt('digg_10/injected_all_10%.csv', synthetic_test, fmt='%d')
