# -*- coding:utf-8 -*-

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='android', help='data name')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--min_epochs', type=int, default=5, help='epochs')

    parser.add_argument('--in_feats', type=int, default=64, help='input dimension')
    parser.add_argument('--h_feats', type=int, default=64, help='h_feats')
    parser.add_argument('--out_feats', type=int, default=64, help='output feats')

    parser.add_argument('--user_dim', type=int, default=32, help='')
    parser.add_argument('--message_dim', type=int, default=32, help='')
    parser.add_argument('--num_users', type=int, default=0, help='')
    parser.add_argument('--num_messages', type=int, default=0, help='')

    parser.add_argument('--heads', type=int, default=4, help='attention heads')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=100, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--patience', type=int, default=30, help='')

    parser.add_argument('--rho', type=int, default=0.001, help='')

    args = parser.parse_args()

    return args