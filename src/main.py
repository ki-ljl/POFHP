# -*- coding:utf-8 -*-

import os
import sys
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from args import args_parser
import os
from get_data import get_data, load_data

from util import train, setup_seed


def main(args):
    msle, male, smape = train(args, graph, pof_graph)
    print('test msle:', msle)
    print('test male:', male)
    print('test smape:', smape)


if __name__ == '__main__':
    setup_seed(12345)
    args = args_parser()
    data_names = ['twitter', 'douban', 'android', 'christianity']

    args, graph, pof_graph = load_data(args, path=root_path)

    if args.data_name == "christianity":
        args.in_feats, args.h_feats, args.out_feats = 32, 32, 32
    elif args.data_name == "douban":
        args.in_feats, args.h_feats, args.out_feats = 64, 128, 64
    else:
        args.in_feats, args.h_feats, args.out_feats = 64, 64, 64

    args.num_users = graph['user'].x.size(0)
    args.num_messages = graph['message'].x.size(0)

    print("----------------------Args----------------------")
    print(args)
    print("----------------------Args----------------------")

    print("----------------------Graph----------------------")
    print(graph)
    print("----------------------Graph----------------------")

    print("----------------------POFGraph----------------------")
    print(pof_graph)
    print("----------------------POFGraph----------------------")

    main(args)
