# -*- coding:utf-8 -*-

import copy

import networkx as nx
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
import numpy as np
import torch
import matplotlib
import time

from pytorchtools import EarlyStopping

from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from get_data import get_data
from models import POFHP
from torch_geometric.utils import to_networkx
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.cuda.amp import autocast, GradScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mape(y, pred):
    return np.mean(np.abs((y - pred) / y))


def get_smape(y, pred):
    temp = (np.abs(y) + np.abs(pred)) / 2
    return np.mean(np.abs(y - pred) / temp)


@torch.no_grad()
def test(args, model, graph, pof_graph):
    test_mask = graph['message'].test_mask
    model.eval()
    out, _ = model(graph, pof_graph)
    label = graph['message'].y
    # test
    y = label[test_mask].cpu().numpy()
    pred = out[test_mask].cpu().numpy()

    y, pred = np.log1p(y), np.log1p(pred)

    test_smape = get_smape(y, pred)
    
    test_msle = np.mean((y - pred)**2)
    test_male = np.mean(np.abs(y - pred))

    return test_msle, test_male, test_smape


# 
def convert_graph_to_bf16(graph):
    graph.user_series_features = graph.user_series_features.to(torch.float16)
    graph.timestamps = graph.timestamps.to(torch.float16)
    graph.user_retweet_message_times = graph.user_retweet_message_times.to(torch.float16)
    graph.seq_last_time = graph.seq_last_time.to(torch.float16)
    graph.mask = graph.mask.to(torch.float16)

    graph['user'].x = graph['user'].x.to(torch.float16)
    graph['message'].x = graph['message'].x.to(torch.float16)
    graph['message'].y = graph['message'].y.to(torch.float16)
    graph['message'].first_time = graph['message'].first_time.to(torch.float16)

    return graph


def recover_graph(graph, pof_graph, copy_graph, copy_pof_graph):
    graph.user_series_features = copy_graph.user_series_features
    graph['user'].x = copy_graph['user'].x
    graph['message'].x = copy_graph['message'].x
    pof_graph.x = copy_pof_graph.x

    return graph, pof_graph


def register_grad_hook(model):
    grads = []
    def hook_fn(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_mean = grad_output[0].abs().mean().item()
            grad_max = grad_output[0].abs().max().item()
            grads.append((grad_mean, grad_max))
        return None
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            layer.register_full_backward_hook(hook_fn)
    return grads


def check_grad(model):
    max_grad = None
    for name, param in model.named_parameters():
        if param.grad is not None:
            # .abs()
            if max_grad is None:
                max_grad = param.grad.abs().max().item()
            else:
                max_grad = max(max_grad, param.grad.abs().max().item())
    
    print(f"grad_max={max_grad}")


def check_model_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            print(f"{name} contains NaN")
            return True
    return False


def train(args, graph, pof_graph):
    graph = graph.to(device)
    pof_graph = pof_graph.to(device)

    model = POFHP(args, activation_str='elu').to(args.device)
    is_half = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_function = torch.nn.MSELoss().to(device)

    if args.data_name == "douban":
        graph = convert_graph_to_bf16(graph)
        model.half()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4, eps=1e-3)

    copy_graph, copy_pof_graph = copy.deepcopy(graph), copy.deepcopy(pof_graph)
    # 
    # scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    
    min_val_loss = np.Inf
    best_msle, best_male, best_smape = 0, 0, 0
    min_epochs = args.min_epochs
    train_loss_ = []
    best_model = None
    train_mask = graph['message'].train_mask

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in tqdm(range(args.epochs)):
        epoch_start = time.time()
        graph, pof_graph = recover_graph(graph, pof_graph, copy_graph, copy_pof_graph)

        model.train()
        optimizer.zero_grad()

        target_label = graph['message'].y[train_mask]

        out, cif_loss = model(graph, pof_graph)
        pred = out[train_mask]

        target_label = torch.log1p(target_label)
        pred = torch.log1p(pred)

        loss = loss_function(
            pred, 
            target_label
        ) + args.rho * cif_loss

        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()

        # check
        check_model_nan(model)
        check_grad(model)

        # validation
        # scheduler.step()
        
        graph, pof_graph = recover_graph(graph, pof_graph, copy_graph, copy_pof_graph)
        msle, male, smape = test(args, model, graph, pof_graph)
        if epoch + 1 > min_epochs and loss.item() < min_val_loss:
            min_val_loss = loss.item()
            best_msle, best_male, best_smape = msle, male, smape
            best_model = copy.deepcopy(model)

        train_loss_.append(loss.item())

        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        epoch_end = time.time()
        cur_lr = optimizer.param_groups[0]['lr']
        # Time Cost: {(epoch_end - epoch_start):04f}S
        print(f'Epoch: {epoch:03d} lr: {cur_lr} Train Loss: {loss.item():.8f}, '
              f'Test msle: {msle:.4f}, male: {male:.4f}, smape: {smape:.4f}')
    
    # save
    state = {'model': best_model.state_dict()}
    torch.save(state, 'ckpt/ckpt_' + args.data_name + '.pt')

    return best_msle, best_male, best_smape
