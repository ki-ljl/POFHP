# -*- coding:utf-8 -*-

import collections
import copy
import pickle
import random

import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import bipartite_subgraph, to_undirected, add_self_loops
from torch_geometric.transforms import RandomNodeSplit
from tqdm import tqdm


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def align_lists(matrix, data_type, t_o=None):
    # cascade or timestamp
    max_length = max(len(row) for row in matrix)
    if data_type == "cascade":
        aligned_matrix = [
            row + [0] * (max_length - len(row))  # pad 0
            for row in matrix
        ]
        user_series = torch.tensor(aligned_matrix)
        return aligned_matrix, user_series
    else:
        min_timestamp = min(min(lst) for lst in matrix)
        matrix = [[timestamp - min_timestamp for timestamp in lst] for lst in matrix]
        t_o = t_o - min_timestamp
        # get mask
        aligned_matrix = [
            row + [-1] * (max_length - len(row))
            for row in matrix
        ]
        _matrix = torch.tensor(aligned_matrix)
        mask = torch.ones_like(_matrix)
        mask = torch.where(_matrix == -1, torch.tensor(-5e9, device=_matrix.device), mask)
        # normalize
        aligned_matrix = [
            row + [0] * (max_length - len(row))
            for row in matrix
        ]
        global_min = min(min(lst) for lst in aligned_matrix if lst)
        global_max = max(max(lst) for lst in aligned_matrix if lst)
        global_min = min(global_min, t_o)
        global_max = max(global_max, t_o)
        # print(global_min, global_max, t_o)
        aligned_matrix = [[(value - global_min) / (global_max - global_min) for value in lst] for lst in aligned_matrix]
        t_o = (t_o - global_min) / (global_max - global_min)
        return aligned_matrix, mask, t_o


def find_last_below_threshold(data, threshold):
    result = []
    for row in data:
        if row[0] > threshold:
            result.append(0)
        elif row[-1] <= threshold:
            result.append(len(row))
        else:
            for index, value in enumerate(row):
                position = index
                if value <= threshold:
                    continue
                else:
                    break
            result.append(position)

    return result


def get_social_hetero_graph(args, cascades, timestamps, all_users, edge_index):
    users_dict = dict(zip(all_users, [idx for idx in range(len(all_users))]))
    # user
    cascades = [[users_dict[x] for x in y] for y in cascades]
    cascades = [x for x in cascades if len(x) > 5]
    timestamps = [x for x in timestamps if len(x) > 5]
    # 
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    cascades = [cascades[i] for i in order]
    timestamps = [timestamps[i] for i in order]
    # set t_o and t_p
    flat_list = [item for sublist in timestamps for item in sublist]
    t_o = np.median(flat_list)  # median time
    # 
    label_idx = find_last_below_threshold(timestamps, threshold=t_o)
    labels = [len(timestamp) - idx for timestamp, idx in zip(timestamps, label_idx)]

    delete_idx = [index for index, value in enumerate(labels) if value == 0 or value == len(timestamps[index])]
    # delete_idx = [index for index, value in enumerate(labels) if value == len(timestamps[index])]
    cascades = [row for idx, row in enumerate(cascades) if idx not in delete_idx]
    timestamps = [row for idx, row in enumerate(timestamps) if idx not in delete_idx]
    label_idx = [row for idx, row in enumerate(label_idx) if idx not in delete_idx]
    labels = [row for idx, row in enumerate(labels) if idx not in delete_idx]

    num_users = len(all_users)
    edge_index = [[users_dict[x] for x in edge_index[0]],
                  [users_dict[x] for x in edge_index[1]]]
    user_edge_index = torch.tensor(edge_index)

    cascades = [cascade[:label_idx[item_id]] for item_id, cascade in enumerate(cascades)]
    timestamps = [timestamp[:label_idx[item_id]] for item_id, timestamp in enumerate(timestamps)]

    seq_last_idx = torch.tensor([len(x) - 1 for x in cascades])
    cascades, user_series = align_lists(cascades, data_type="cascade")
    timestamps, mask, t_o = align_lists(timestamps, data_type="timestamp", t_o=t_o)
    seq_last_time = [timestamp[idx] for timestamp, idx in zip(timestamps, seq_last_idx)]

    # construct graph
    num_messages = len(cascades)
    user_to_message_edge_index = [[], []]
    message_to_user_edge_index = [[], []]
    user_propagate_user_edge_index = [[], []]
    # 
    user_retweet_message_times = []
    for item_id, users in enumerate(cascades):
        users = users[:label_idx[item_id]]
        for idx, user in enumerate(users):
            if idx != 0:
                user_propagate_user_edge_index[0].append(users[idx - 1])
                user_propagate_user_edge_index[1].append(users[idx])

            user_retweet_message_times.append(timestamps[item_id][idx])
            user_to_message_edge_index[0].append(user)
            user_to_message_edge_index[1].append(item_id)
            message_to_user_edge_index[0].append(item_id)
            message_to_user_edge_index[1].append(user)

    user_to_message_edge_index = torch.tensor(user_to_message_edge_index)
    message_to_user_edge_index = torch.tensor(message_to_user_edge_index)
    user_propagate_user_edge_index = torch.tensor(user_propagate_user_edge_index)
    # user_to_item_edge_index, _ = add_self_loops(user_to_item_edge_index)
    #
    graph = HeteroData()
    graph['user'].x = torch.randn(num_users, args.user_dim)
    graph['message'].x = torch.randn(num_messages, args.message_dim)
    graph['message'].y = torch.FloatTensor(labels)

    num_of_per_cascade = [1 for _ in range(num_messages)]

    total_nums = np.sum(num_of_per_cascade)
    seq_idx = []
    for i in range(total_nums):
        for j in range(num_of_per_cascade[i]):
            seq_idx.append(i)
    seq_idx = torch.tensor(seq_idx)
    #
    start_idx = 0
    message_first_time = []
    for i in range(num_messages):
        if i != 0:
            start_idx = start_idx + num_of_per_cascade[i - 1]
        cur_first_time = timestamps[start_idx][0]
        for j in range(num_of_per_cascade[i]):
            if timestamps[start_idx + j][0] < cur_first_time:
                cur_first_time = timestamps[start_idx + j][0]

        message_first_time.append(cur_first_time)

    graph.user_series = user_series
    graph.user_series_features = graph['user'].x[graph.user_series]
    graph.seq_idx = seq_idx
    graph.timestamps = torch.FloatTensor(timestamps)
    message_first_time = torch.tensor(message_first_time)
    graph.t_o = t_o
    graph.num_of_per_cascade = torch.tensor(num_of_per_cascade)
    graph.user_retweet_message_times = torch.FloatTensor(user_retweet_message_times)
    graph['message'].first_time = message_first_time
    # graph.seq_first_time = message_first_time[seq_idx]
    graph.seq_last_time = torch.tensor(seq_last_time)
    graph.seq_last_idx = seq_last_idx
    graph.mask = mask
    # supervised mask
    train_len = int(num_messages * 0.8)

    idx = np.arange(num_messages)
    train_mask = np.zeros(num_messages)
    train_mask[idx[:train_len]] = 1
    graph['message'].train_mask = torch.BoolTensor(train_mask)

    test_mask = np.zeros(num_messages)
    test_mask[idx[train_len:]] = 1
    graph['message'].test_mask = torch.BoolTensor(test_mask)

    # edge_index
    graph['user', 'friendship', 'user'].edge_index = user_edge_index
    graph['user', 'to', 'message'].edge_index = user_to_message_edge_index
    graph['message', 'to', 'user'].edge_index = message_to_user_edge_index
    graph['user', 'pr', 'user'].edge_index = user_propagate_user_edge_index

    graph['user', 'friendship', 'user'].edge_index = to_undirected(graph['user', 'friendship', 'user'].edge_index)
    graph['user', 'friendship', 'user'].edge_index, _ = add_self_loops(graph['user', 'friendship', 'user'].edge_index)

    print('train val test:', torch.sum(graph['message'].train_mask), torch.sum(graph['message'].test_mask))

    return graph


def count_roots(A):
    counter = {}
    for root, _ in A:
        if root in counter:
            counter[root] += 1
        else:
            counter[root] = 1
    max_root = max(counter.keys()) if counter else 0

    B = [counter.get(i, 0) for i in range(max_root + 1)]
    return B


def get_stackexchange_hetero_graph(args, se_cascades, se_timestamps, all_users, edge_index):
    users_dict = dict(zip(all_users, [idx for idx in range(len(all_users))]))

    cascades, timestamps = [], []
    for k, v in se_cascades.items():
        se_cascades[k] = [[users_dict[x] for x in y] for y in v]
        se_cascades[k] = [x for x in se_cascades[k] if len(x) > 5]
        se_timestamps[k] = [x for x in se_timestamps[k] if len(x) > 5]
        # 
        sub_cascades = [(k, users) for users in se_cascades[k]]
        cascades.extend(sub_cascades)
        sub_timestamps = [(k, times) for times in se_timestamps[k]]
        timestamps.extend(sub_timestamps)

    # 
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1][1])]
    cascades = [cascades[i] for i in order]
    timestamps = [timestamps[i] for i in order]

    flat_list = [item for _, sublist in timestamps for item in sublist]
    t_o = np.median(flat_list)
    only_timestamps = [timestamp for _, timestamp in timestamps]
    label_idx = find_last_below_threshold(only_timestamps, threshold=t_o)
    labels = [len(timestamp) - idx for timestamp, idx in zip(only_timestamps, label_idx)]

    delete_idx = [index for index, value in enumerate(labels) if value == 0 or value == len(only_timestamps[index])]
    cascades = [row for idx, row in enumerate(cascades) if idx not in delete_idx]
    timestamps = [row for idx, row in enumerate(timestamps) if idx not in delete_idx]
    labels = [row for idx, row in enumerate(labels) if idx not in delete_idx]
    label_idx = [row for idx, row in enumerate(label_idx) if idx not in delete_idx]

    num_users = len(all_users)
    edge_index = [[users_dict[x] for x in edge_index[0]],
                  [users_dict[x] for x in edge_index[1]]]
    user_edge_index = torch.tensor(edge_index)

    # construct graph
    only_root = [root for root, _ in timestamps]
    only_root = set(only_root)
    root_dict = dict(zip(only_root, [x for x in range(len(only_root))]))

    cascades = [(root_dict[root], cascade) for root, cascade in cascades]
    timestamps = [(root_dict[root], timestamp) for root, timestamp in timestamps]
    order = [i[0] for i in sorted(enumerate(cascades), key=lambda x: x[1][0])]
    cascades = [cascades[idx] for idx in order]
    timestamps = [timestamps[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    label_idx = [label_idx[idx] for idx in order]

    cascades = [(root, cascade[:label_idx[idx]]) for idx, (root, cascade) in enumerate(cascades)]
    timestamps = [(root, timestamp[:label_idx[idx]]) for idx, (root, timestamp) in enumerate(timestamps)]

    only_cascades = [cascade for _, cascade in cascades]
    only_timestamps = [timestamp for _, timestamp in timestamps]
    seq_last_idx = torch.tensor([len(x) - 1 for x in only_cascades])
    only_cascades, user_series = align_lists(only_cascades, data_type="cascade")
    only_timestamps, mask, t_o = align_lists(only_timestamps, data_type="timestamp", t_o=t_o)
    seq_last_time = [timestamp[idx] for timestamp, idx in zip(only_timestamps, seq_last_idx)]

    num_messages = len(only_root)
    user_to_message_edge_index = [[], []]
    message_to_user_edge_index = [[], []]
    user_propagate_user_edge_index = [[], []]
    # 
    user_retweet_message_times = []
    for idx, (item_id, users) in enumerate(cascades):
        # 
        users = users[:label_idx[idx]]
        for sub_idx, user in enumerate(users):
            if sub_idx != 0:
                user_propagate_user_edge_index[0].append(users[sub_idx - 1])
                user_propagate_user_edge_index[1].append(users[sub_idx])

            user_retweet_message_times.append(only_timestamps[idx][sub_idx])
            user_to_message_edge_index[0].append(user)
            user_to_message_edge_index[1].append(item_id)
            message_to_user_edge_index[0].append(item_id)
            message_to_user_edge_index[1].append(user)

    user_to_message_edge_index = torch.tensor(user_to_message_edge_index)
    message_to_user_edge_index = torch.tensor(message_to_user_edge_index)
    user_propagate_user_edge_index = torch.tensor(user_propagate_user_edge_index)
    #
    graph = HeteroData()
    #
    graph['user'].x = torch.randn(num_users, args.user_dim)
    graph['message'].x = torch.randn(num_messages, args.message_dim)

    max_len = len(only_cascades[0])
    # print([root for root, _ in cascades])
    num_of_per_cascade = count_roots(cascades)
    # 
    total_nums = np.sum(num_of_per_cascade)
    seq_idx = []
    for i in range(len(only_root)):
        for j in range(num_of_per_cascade[i]):
            seq_idx.append(i)
    seq_idx = torch.tensor(seq_idx)
    # 
    new_labels = []
    print(np.sum(num_of_per_cascade), len(num_of_per_cascade))
    message_first_time = []
    start_idx = 0
    for i in range(num_messages):
        if i != 0:
            start_idx = start_idx + num_of_per_cascade[i - 1]
        cur_first_time = only_timestamps[start_idx][0]
        new_labels.append(np.sum(labels[start_idx:start_idx + num_of_per_cascade[i]]))
        for j in range(num_of_per_cascade[i]):
            if only_timestamps[start_idx + j][0] < cur_first_time:
                cur_first_time = only_timestamps[start_idx + j][0]

        message_first_time.append(cur_first_time)

    graph.user_series = user_series
    graph.user_series_features = graph['user'].x[graph.user_series]
    graph.seq_idx = seq_idx
    graph.timestamps = torch.FloatTensor(only_timestamps)
    message_first_time = torch.tensor(message_first_time, dtype=torch.float32)
    graph.t_o = t_o
    graph.num_of_per_cascade = torch.tensor(num_of_per_cascade)
    graph.user_retweet_message_times = torch.FloatTensor(user_retweet_message_times)
    graph['message'].y = torch.FloatTensor(new_labels)
    graph['message'].first_time = message_first_time
    # graph.seq_first_time = message_first_time[seq_idx]
    graph.seq_last_time = torch.tensor(seq_last_time)
    graph.seq_last_idx = seq_last_idx
    graph.mask = mask
    # mask
    train_len = int(num_messages * 0.8)

    idx = np.arange(num_messages)
    train_mask = np.zeros(num_messages)
    train_mask[idx[:train_len]] = 1
    graph['message'].train_mask = torch.BoolTensor(train_mask)

    test_mask = np.zeros(num_messages)
    test_mask[idx[train_len:]] = 1
    graph['message'].test_mask = torch.BoolTensor(test_mask)

    graph['user', 'friendship', 'user'].edge_index = user_edge_index
    graph['user', 'to', 'message'].edge_index = user_to_message_edge_index
    graph['message', 'to', 'user'].edge_index = message_to_user_edge_index
    graph['user', 'pr', 'user'].edge_index = user_propagate_user_edge_index

    graph['user', 'friendship', 'user'].edge_index = to_undirected(graph['user', 'friendship', 'user'].edge_index)
    graph['user', 'friendship', 'user'].edge_index, _ = add_self_loops(graph['user', 'friendship', 'user'].edge_index)

    print('train val test:', torch.sum(graph['message'].train_mask), torch.sum(graph['message'].test_mask))

    return graph


def get_data(args):
    data_name = args.data_name

    with open('data/' + data_name + '/edges.txt', 'r') as file:
        edges_lines = file.readlines()

    all_users = set()
    #
    edge_index = [[], []]
    for line in edges_lines:
        line = line.strip().split(',')
        all_users.add(line[0])
        all_users.add(line[1])
        edge_index[0].append(line[0])
        edge_index[1].append(line[1])

    #
    data = open('data/' + data_name + '/cascades.txt', 'r')
    data = data.readlines()
    timestamps = []
    cascades = []
    se_timestamps = collections.defaultdict(list)  # android and chris
    se_cascades = collections.defaultdict(list)
    # 
    for f in tqdm(data):
        if len(f.strip()) == 0:
            continue
        f = f.strip().split(',')
        sub_times = []
        sub_cascades = []
        for c in f:
            if data_name == 'douban' or data_name == 'twitter':
                if len(c.split()) != 2:
                    continue
                else:
                    # user1 time1 user time2...
                    user, timestamp = c.split()
            else:
                # root user1 time1 user2 time2...
                if len(c.split()) == 2:
                    user, timestamp = c.split()
                elif len(c.split()) == 3:
                    root, user, timestamp = c.split()
                else:
                    continue
            # 
            all_users.add(user)
            sub_cascades.append(user)
            sub_times.append(float(timestamp))

        if data_name == 'douban' or data_name == 'twitter':
            cascades.append(sub_cascades)
            timestamps.append(sub_times)
        else:
            se_cascades[root].append(sub_cascades)
            se_timestamps[root].append(sub_times)

    if data_name == 'douban' or data_name == 'twitter':
        graph = get_social_hetero_graph(args, cascades, timestamps, all_users, edge_index)
    else:
        graph = get_stackexchange_hetero_graph(args, se_cascades, se_timestamps, all_users, edge_index)

    pof_graph = get_pof(graph)
    user_retweet_message_times = graph.user_retweet_message_times
    graph.user_retweet_message_times = torch.cat((user_retweet_message_times, user_retweet_message_times))
    print(pof_graph)
    print(graph)
    return graph, pof_graph


def get_pof(graph):
    # 
    num_messages = graph['message'].x.size(0)
    num_users = graph['user'].x.size(0)
    um_edge_index = graph['user', 'to', 'message'].edge_index.t().numpy().tolist()
    edge_index = [[], []]

    for user, message in um_edge_index:
        user_idx, message_idx = user, message + num_users
        edge_index[0].append(user_idx)
        edge_index[1].append(message_idx)
        #
        edge_index[0].append(message_idx)
        edge_index[1].append(user_idx)

    pof_graph = Data(edge_index=torch.tensor(edge_index))

    return pof_graph


def load_data(args, path):
    graph = load_pickle(path + '/data/' + args.data_name + '/graph.pkl')
    pof_graph = load_pickle(path + '/data/' + args.data_name + '/pof_graph.pkl')
    user_emb = load_pickle(path + '/data/' + args.data_name + '/user_emb.pkl')
    message_emb = load_pickle(path + '/data/' + args.data_name + '/message_emb.pkl')
    graph['user'].x = user_emb
    graph['message'].x = message_emb
    graph.user_series_features = graph['user'].x[graph.user_series]
    args.user_dim = user_emb.size(1)
    args.message_dim = message_emb.size(1)

    return args, graph, pof_graph

