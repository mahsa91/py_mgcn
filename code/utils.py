import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score

import random
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_bet_adj(layer_num1, layer_num2, idx_map_l1, idx_map_l2, path="../data/cora/", dataset="cora"):
    temp = "{}{}.bet" + str(layer_num1) + "_" + str(layer_num2)
    # print("bet file")
    # print(temp)
    if not isfile(temp.format(path, dataset)):
        return None
    edges_unordered = np.genfromtxt(temp.format(path, dataset), dtype=np.int32)
    # idx1 = np.array(np.unique(edges_unordered[:,0]), dtype=np.int32)
    # idx2 = np.array(np.unique(edges_unordered[:, 1]), dtype=np.int32)
    # idx_map_l1 = {j: i for i, j in enumerate(idx1)}
    # idx_map_l2 = {j: i for i, j in enumerate(idx2)}
    N1 = len(list(idx_map_l1))
    N2 = len(list(idx_map_l2))
    edges = np.array(list(map(idx_map_l1.get, edges_unordered[:,0])) + list(map(idx_map_l2.get, edges_unordered[:,1])),
                     dtype=np.int32).reshape(edges_unordered.shape, order='F')
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N1, N2),
                        dtype=np.float32)
    adj_orig = sparse_mx_to_torch_sparse_tensor(adj)

    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj_orig


def load_in_adj(layer_num, idx_map=None, path="../data/cora/", dataset="cora"):
    temp = "{}{}.adj" + str(layer_num)
    edges_unordered = np.genfromtxt(temp.format(path, dataset),
                                    dtype=np.int32)
    if idx_map is None:
        idx = np.array(np.unique(edges_unordered.flatten()), dtype=np.int32)
        N = len(list(idx))
        idx_map = {j: i for i, j in enumerate(idx)}
    else:
        N = len(list(idx_map))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N),
                        dtype=np.float32)
    # print("Edges")
    # print(np.count_nonzero(adj.toarray()))
    #build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_orig = sparse_mx_to_torch_sparse_tensor(adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj_orig


def load_features_labels(layer_num, path, dataset,N=-1):
    print('Loading {} dataset...'.format(dataset))
    temp = "{}{}.feat"+str(layer_num)
    idx_features_labels = np.genfromtxt(temp.format(path, dataset),
                                        dtype=np.dtype(str))
    temp = idx_features_labels[:, 1:-1]
    if temp.size == 0:
        features = sp.csr_matrix(np.identity(N), dtype=np.float32)
    else:
        features = sp.csr_matrix(temp, dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    #build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    features = normalize(features)

    features = sp.csr_matrix(features)
    features = sparse_mx_to_torch_sparse_tensor(features)
    # features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return features, labels, idx_map


def train_val_test_split(N, val_size=0.2, test_size=0.2, random_state=1):
    idx_train_temp, idx_test = train_test_split(range(N), test_size=test_size, random_state=random_state)
    if val_size == 0:
        idx_train = idx_train_temp
    else:
        idx_train, idx_val = train_test_split(idx_train_temp, test_size=val_size, random_state=random_state)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test


def trains_vals_tests_split(n_layers, labels_sizes, val_size, test_size, random_state):
    idx_trains = []
    idx_vals = []
    idx_tests = []
    for i in range(n_layers):
        idx_train, idx_val, idx_test = train_val_test_split(labels_sizes[i], val_size, test_size, random_state)
        idx_trains.append(idx_train)
        idx_vals.append(idx_val)
        idx_tests.append(idx_test)

    return idx_trains, idx_vals, idx_tests


def load_data(path="../data/cora/", dataset="cora"):
    layers = 0
    adjs = []
    adjs_orig = []
    adjs_sizes = []
    adjs_pos_weights = []
    adjs_norms = []

    bet_adjs = []
    bet_adjs_orig = []
    bet_adjs_sizes = []
    bet_pos_weights = []
    bet_norms = []

    features = []
    features_sizes = []
    labels = []
    labels_nclass = []

    idx_maps = []
    for f in listdir(path):
        if isfile(join(path, f)):
            if 'adj' in f:
                layers += 1
    for i in range(layers):
        feature, label, idx_map = load_features_labels(i, path, dataset)
        adj, adj_orig = load_in_adj(i, idx_map, path, dataset)

        pos_weight = float(adj.shape[0] * adj.shape[1] - adj.to_dense().sum()) / adj.to_dense().sum()
        norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.to_dense().sum()) * 2)

        idx_maps.append(idx_map)

        adjs.append(adj)
        adjs_orig.append(adj_orig)
        adjs_sizes.append(tuple(adj.size()))
        adjs_pos_weights.append(pos_weight)
        adjs_norms.append(norm)
        features.append(feature)
        features_sizes.append(feature.shape[1])
        labels.append(label)
        labels_nclass.append((label.max().item() + 1))

    for i in range(layers):
        for j in range(i+1,layers):
            bet_adj, bet_adj_orig = load_bet_adj(i, j, idx_maps[i], idx_maps[j], path, dataset)
            bet_adjs.append(bet_adj)
            bet_adjs_orig.append(bet_adj_orig)
            bet_adjs_sizes.append(tuple(bet_adj.size()) if not bet_adj is None else tuple((0,0)))
            if not bet_adj is None:
                pos_weight = float(
                    bet_adj.shape[0] * bet_adj.shape[1] - bet_adj.to_dense().sum()) / bet_adj.to_dense().sum()
                norm = bet_adj.shape[0] * bet_adj.shape[1] / float((bet_adj.shape[0] * bet_adj.shape[1] - bet_adj.to_dense().sum()) * 2)
            else:
                norm = None
                pos_weight = None
            bet_pos_weights.append(pos_weight)
            bet_norms.append(norm)

    return adjs, adjs_orig, adjs_sizes, adjs_pos_weights, adjs_norms, bet_pos_weights, bet_norms, bet_adjs, bet_adjs_orig, \
           bet_adjs_sizes, features, features_sizes, labels, labels_nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def class_accuracy(output, labels, type=None):
    preds = output.max(1)[1].type_as(labels)
    return accuracy_score(labels.data, preds)


def class_f1(output, labels, type='micro'):
    preds = output.max(1)[1].type_as(labels)
    return f1_score(labels.data, preds, average=type)


def layer_accuracy(output, real, type=None):
    preds = output.data.clone()
    true = real.data.clone()
    preds[output < 0.5] = 0
    preds[output >= 0.5] = 1
    true[real > 0] = 1
    return accuracy_score(true, preds)


def layer_f1(output, real, type='micro'):
    preds = output.data.clone()
    true = real.data.clone()
    preds[output < 0.5] = 0
    preds[output >= 0.5] = 1
    true[real > 0] = 1
    return f1_score(true, preds, average=type)


def writer_data(values, writer, epoch, type, name):
    try:
        for i, j in enumerate(values):
            # my_dic[name+str(i)] = j
            writer.add_scalar(type + "/" + name + str(i), j, epoch)
    except TypeError as te:
        writer.add_scalar(type + "/" + name, values, epoch)


# type = in_class, in_struc, bet_struc
def dict_to_writer(stats, writer, epoch, type, train_test_val):
    for key, value in stats.items():
        # type_str = train_test_val + "/" + type + '/Loss' if key == 'loss' else train_test_val + type + '/Stats'
        type_str = train_test_val + "/" + type
        writer_data(value, writer, epoch, type_str, key)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def between_index(n_inputs, i,j):
    return int(i * n_inputs - (i * (i + 1) / 2) + (j - i - 1))

def gather_edges(pos_edges, neg_edges):
    all_edges = [[], []]
    all_edges[0].extend([idx_i[0] for idx_i in pos_edges])
    all_edges[1].extend([idx_i[1] for idx_i in pos_edges])
    all_edges[0].extend([idx_i[0] for idx_i in neg_edges])
    all_edges[1].extend([idx_i[1] for idx_i in neg_edges])
    return all_edges


