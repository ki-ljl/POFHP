# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch_scatter
import gc

from torch import nn
import torch.nn.functional as F
import torch.sparse as sparse
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

from args import args_parser
from get_data import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gain = nn.init.calculate_gain('relu')


def get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_str == 'elu':
        return nn.ELU()
    elif activation_str == 'prelu':
        return nn.PReLU()
    elif activation_str == 'silu':
        return nn.SiLU()
    elif activation_str == 'gelu':
        return nn.GELU()
    elif activation_str == 'tanh':
        return nn.Tanh()
    elif activation_str == 'softplus':
        return nn.Softplus()
    elif activation_str == 'softsign':
        return nn.Softsign()
    else:
        raise ValueError("Unsupported activation function: " + activation_str)


class POFHPConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_heads: int = 4,
            bias: bool = True,
            activation_str='elu',
            is_act=True,
            **kwargs,
    ):
        """
        Args:
        """
        super(POFHPConv, self).__init__()
        #
        self.beta = 1.0
        self.is_act = is_act
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        # cif
        self.w_his = nn.Linear(self.out_channels, 1, bias=False)
        self.w_intra = nn.Linear(self.out_channels, 1, bias=False)
        self.w_inter = nn.Linear(self.out_channels, 1, bias=False)
        self.w_base = nn.Linear(self.out_channels, 1, bias=False)
        self.gru = nn.GRU(
            input_size=self.out_channels, 
            hidden_size=self.out_channels,
            batch_first=True, num_layers=1
        )

        self.activation = get_activation(activation_str)
        self.soft_plus = nn.Softplus(beta=self.beta)
        # att
        self.poc_att = nn.Parameter(torch.zeros(size=(1, 2 * self.out_channels)))
        self.interest_att = nn.Parameter(torch.zeros(size=(1, 2 * self.out_channels)))

        # nn.init.xavier_uniform_(self.poc_att.data, gain=gain)
        # nn.init.xavier_uniform_(self.interest_att.data, gain=gain)
        nn.init.kaiming_normal_(self.poc_att.data)
        nn.init.kaiming_normal_(self.interest_att.data)

        self.short_u_p_u_att = GATConv(out_channels, out_channels, heads=n_heads, concat=False)
        self.friend_att = GATConv(out_channels, out_channels, heads=n_heads, concat=False)
        # transformer
        self.Q = nn.Linear(out_channels, out_channels)
        self.K = nn.Linear(out_channels, out_channels)
        self.V = nn.Linear(out_channels, out_channels)
        self.f_time = nn.Sequential(
            nn.Linear(1, out_channels)
        )

    def decrease_function(self, x):
        x.mul_(-2.0).exp_()
        return x
        # return torch.exp(-2.0 * x)
    
    def sparse_dropout(self, x: torch.Tensor, p=0.5):
        x = x.coalesce()
        return torch.sparse_coo_tensor(x.indices(),
                                       F.dropout(x.values(),
                                                 p=p,
                                                 training=self.training), size=x.size())

    def get_pofe(self, graph):
        # cif and pofe
        user_series_features = graph.user_series_features
        timestamps = graph.timestamps  # total_num, max_len
        # gru
        user_series_h, _ = self.gru(user_series_features)  # total_num max_len, out_channels
        # 
        num_seqs, seq_len, dim = user_series_h.shape
        x_mk = graph['message'].x
        # broadcast
        x_mk_broadcast = x_mk[graph.seq_idx]  # num_seqs out_channels
        # 
        cif = torch.zeros(num_seqs, seq_len).to(x_mk.device)
        X = torch.zeros(num_seqs, seq_len).to(x_mk.device)

        # 1. history
        time_diff_matrix = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [batch, seq, seq]
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=-1).bool().to(timestamps.device)  # [seq, seq]

        weights_all = self.w_his(user_series_h).squeeze(-1)  # [batch, seq]
        masked_weights = weights_all.unsqueeze(1) * mask.unsqueeze(0)  # [batch, seq, seq]
        
        X_all = (self.decrease_function(time_diff_matrix) * masked_weights).sum(dim=2)  # [batch, seq]
        X[:, 1:] = X_all[:, 1:]

        cif = cif + torch.sigmoid(X)
        # 2. intra
        X.fill_(0)
        num_messages = graph['message'].x.size(0)
        start_idx = 0
        for i in range(num_messages):
            if i != 0:
                start_idx += graph.num_of_per_cascade[i - 1]
            if graph.num_of_per_cascade[i] == 1:
                continue

            end_idx = start_idx + graph.num_of_per_cascade[i]
            # 
            cur_features = user_series_h[start_idx:end_idx, :, :]  # num seq dim
            cur_features = self.w_intra(cur_features).squeeze(-1)  # num seq
            # last time
            cur_last_time = graph.seq_last_time[start_idx:end_idx]
            # time_diff[i, j] = cur_last_time[i] - cur_last_time[j]
            time_diff = cur_last_time.unsqueeze(1) - cur_last_time.unsqueeze(0)
            time_diff = self.decrease_function(time_diff)
            result = time_diff.to(cur_features.dtype) @ cur_features
            X[start_idx:end_idx] = result

        cif = cif + torch.sigmoid(X)
        # 3. inter
        X.fill_(0)
        message_first_time = graph['message'].first_time  # num_message\
        # Compute the term (T[i, j] - F[k])
        diff = timestamps.unsqueeze(-1) - message_first_time.view(1, 1, -1)  # num_seqs, seq_len, num_message
        diff = self.decrease_function(diff)
        X = torch.einsum("abc,cd->abd", diff, x_mk)  # num_seqs, seq_len dim
        X = X - x_mk_broadcast.unsqueeze(1)
        X = self.w_inter(X).squeeze(-1)

        cif = cif + torch.sigmoid(X)
        # 4. base
        X = self.w_base(x_mk_broadcast.unsqueeze(1).repeat(1, seq_len, 1)).squeeze(-1)  # num_seqs, seq_len
        
        cif = cif + torch.sigmoid(X)
        
        # 5. add and softplus
        cif = self.soft_plus(cif)  # num_seqs, seq_len
        # energy
        # last time
        selected_cif = torch.gather(cif, dim=1, index=graph.seq_last_idx.unsqueeze(1)).squeeze(1)  # num_seqs
        
        pofe = scatter(selected_cif, graph.seq_idx, dim=0, reduce='sum')
        return cif, pofe

    def pof_learning(self, graph, pof_graph):
        # pof learning
        num_messages = graph['message'].x.size(0)
        x, edge_index = pof_graph.x, pof_graph.edge_index
        edge_h = torch.cat((x[edge_index[0, :], :], x[edge_index[1, :], :]),
                           dim=1).t()
        values = self.poc_att.mm(edge_h).squeeze()

        user_retweet_message_times = graph.user_retweet_message_times  # num_um_edges
        time_obs = torch.ones_like(user_retweet_message_times) * graph.t_o
        time_weight = self.decrease_function(time_obs - user_retweet_message_times)
        values = F.leaky_relu(values) * time_weight
        # softmax
        N = x.size(0)
        sp_edge_h = torch.sparse_coo_tensor(edge_index, -values,
                                            size=(N, N))  # values() = E
                                            
        if sp_edge_h.dtype == torch.float16:                          
            sp_edge_h_float32 = sp_edge_h.to(torch.float32)
            sp_edge_h_float32 = sparse.softmax(sp_edge_h_float32, dim=1)
            h_prime = torch.sparse.mm(sp_edge_h_float32, x.to(torch.float32))  # (NxN) * (NxF') = (NxF')
            h_prime = h_prime.to(torch.float16)
        else:
            sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
            h_prime = torch.sparse.mm(sp_edge_h, x)  # (NxN) * (NxF') = (NxF')
        
        if self.is_act:
            h_prime = self.activation(h_prime)

        return h_prime[-num_messages:, :]

    def user_prop_conv(self, graph):
        # 1. short dependency
        x = graph['user'].x
        edge_index = graph['user', 'pr', 'user'].edge_index
        short_x = self.short_u_p_u_att(x, edge_index)  # num_users, out_channels
        
        # 2. long dependency
        H = graph.user_series_features  # num seq_len dim
        N, S, D = H.size()

        repeat_message_x = graph['message'].x[graph.seq_idx]  # num dim
        
        H = torch.cat((repeat_message_x.unsqueeze(1), H), dim=1)  # num seq_len + 1 dim
        
        timestamps = self.f_time(graph.timestamps.unsqueeze(-1))  # num seq_len d
        timestamps = torch.cat((torch.zeros((N, 1, D), device=timestamps.device, dtype=timestamps.dtype), timestamps), dim=1)
        
        Q, K, V = self.Q(H + timestamps), self.K(H + timestamps), self.V(H + timestamps)
        att = torch.einsum("nsd,nda->nsa", Q, K.permute(0, 2, 1))
        att = att / np.sqrt(self.out_channels)  # n s s

        mask = torch.cat((torch.ones((N, 1), device=graph.mask.device, dtype=graph.mask.dtype), graph.mask), dim=1)  # n s

        att = torch.where(mask.unsqueeze(1) == 1, att, -float("inf"))
        att = F.softmax(att, dim=-1)
        h_prime = torch.einsum("nsa,nad->nsd", att, V)

        long_message_x = torch_scatter.scatter(h_prime[:, 0, :], graph.seq_idx, dim=0, reduce='mean')

        if self.is_act:
            short_x = self.activation(short_x)
            long_message_x = self.activation(long_message_x)

        return short_x, long_message_x

    def user_interest_conv(self, graph, pof_graph, pofe, message_x):
        x, edge_index = pof_graph.x, pof_graph.edge_index
        edge_index, _ = add_self_loops(edge_index)
        num_messages = graph['message'].x.size(0)
        num_users = graph['user'].x.size(0)
        # 1. normalize
        init_message_x = x[-num_messages:, :]  # num_messages, dim
        pofe = pofe / torch.sum(pofe, dim=-1)

        new_message_x = init_message_x + pofe.unsqueeze(-1) * message_x
        # 
        edge_h = torch.cat((x[edge_index[0, :], :], x[edge_index[1, :], :]), dim=1).t()
        values = self.interest_att.mm(edge_h).squeeze()
        N = x.size(0)
        sp_edge_h = torch.sparse_coo_tensor(edge_index, -F.leaky_relu(values),
                                            size=(N, N))  # values() = E
        
        # float16 or float32
        if x.dtype == torch.float16:
            sp_edge_h_float32 = sp_edge_h.to(torch.float32)
            sp_edge_h_float32 = sparse.softmax(sp_edge_h_float32, dim=1)
            # sp_edge_h_float32 = self.sparse_dropout(sp_edge_h_float32)
            h_prime = sparse.mm(
                sp_edge_h_float32,
                torch.cat((x[:num_users, :].to(torch.float32), new_message_x.to(torch.float32)), dim=0)
            )
            h_prime = h_prime.to(torch.float16)

        else:
            sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
            # sp_edge_h = self.sparse_dropout(sp_edge_h)
            h_prime = sparse.mm(
                sp_edge_h,
                torch.cat((x[:num_users, :], new_message_x), dim=0)
            )

        if self.is_act:
            h_prime = self.activation(h_prime)

        return h_prime[:num_users]

    def friend_conv(self, graph):
        x = graph['user'].x
        edge_index = graph['user', 'friendship', 'user'].edge_index
        x = self.friend_att(x, edge_index)  # num_users, out_channels

        if self.is_act:
            x = self.activation(x)

        return x

    def forward(self, graph, pof_graph):
        # proj
        graph['user'].x = self.proj(graph['user'].x)
        graph['message'].x = self.proj(graph['message'].x)
        pof_graph.x = self.proj(pof_graph.x)
        graph.user_series_features = graph['user'].x[graph.user_series]
        # 1. get pofe
        cif, pofe = self.get_pofe(graph)
        # 2. pof learning
        pof_message_x = self.pof_learning(graph, pof_graph)
        # 3. user propagation user
        user_prop_x, long_message_x = self.user_prop_conv(graph)
        # 4. user interest user
        user_interest_x = self.user_interest_conv(graph, pof_graph, pofe, pof_message_x)
        # 5. user friendship user
        user_friend_x = self.friend_conv(graph)

        # add
        message_x = pof_message_x + long_message_x
        user_x = user_prop_x + user_interest_x + user_friend_x
        
        return cif, message_x, user_x


class POFHP(nn.Module):
    def __init__(self, args, activation_str='elu'):
        super(POFHP, self).__init__()
        self.args = args
        in_channels = args.in_feats
        hidden_channels = args.h_feats
        out_channels = args.out_feats
        user_dim, message_dim = args.user_dim, args.message_dim
        heads = args.heads

        self.conv1 = POFHPConv(in_channels, hidden_channels, n_heads=heads,
                               activation_str=activation_str, is_act=True)
        self.conv2 = POFHPConv(hidden_channels, out_channels, n_heads=heads,
                               activation_str=activation_str, is_act=True)

        self.user_lin = nn.Linear(user_dim, in_channels)
        self.message_lin = nn.Sequential(
            nn.Linear(message_dim, in_channels)
        )

        self.wij_att = nn.Parameter(torch.zeros(size=(1, 2 * out_channels)))
        nn.init.xavier_uniform_(self.wij_att.data, gain=gain)

        # 
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Softplus(),
            nn.Linear(out_channels, out_channels),
            nn.Softplus(),
            nn.Linear(out_channels, 1)
        )
    
    def sparse_dropout(self, x: torch.Tensor, p=0.5):
        x = x.coalesce()
        return torch.sparse_coo_tensor(x.indices(),
                                       F.dropout(x.values(),
                                                 p=p,
                                                 training=self.training), size=x.size())

    def trans_dimensions(self, graph, pof_graph):
        graph.user_series_features = self.user_lin(graph.user_series_features)
        graph['user'].x = self.user_lin(graph['user'].x)
        graph['message'].x = self.message_lin(graph['message'].x)
        # pof graph
        pof_graph.x = torch.cat((graph['user'].x, graph['message'].x), dim=0)

        return graph, pof_graph

    def compute_loss(self, graph, cif):
        # cif num_seqs, seq_len
        # part1
        cif = torch.sigmoid(cif)  #
        cif = torch.log(cif)
        seq_last_idx = graph.seq_last_idx + 1
        row_indices, col_indices = torch.meshgrid(torch.arange(cif.size(0), device=cif.device),
                                                  torch.arange(cif.size(1), device=cif.device), indexing="ij")
        valid_indices = col_indices < seq_last_idx.unsqueeze(1)
        # 
        valid_positions = torch.nonzero(valid_indices)
        # 
        part_one_likelihood = torch.zeros(cif.size(0), device=cif.device)
        part_one_likelihood.scatter_add_(0, valid_positions[:, 0], cif[valid_indices])
        # part2
        # 
        mask = torch.arange(cif.size(1), device=cif.device).expand(cif.size(0), -1) <= seq_last_idx.unsqueeze(1)
        # 
        cif_masked = cif * mask.float()
        row_sums = torch.sum(cif_masked, dim=-1)
        lengths = torch.sum(mask.float(), dim=-1)
        part_two_likelihood = row_sums / lengths
        # sum
        part_one_likelihood = torch.sum(part_one_likelihood, dim=-1)
        part_two_likelihood = torch.sum(part_two_likelihood, dim=-1)

        return -(part_one_likelihood - part_two_likelihood)

    def predict(self, pof_graph, message_x, user_x):
        x = torch.cat((user_x, message_x), dim=0)  # user + message
        edge_index = pof_graph.edge_index  # user-message
        # \tau_{ij}
        e = torch.sigmoid(torch.cov(x))
        covariance_vector = e[edge_index[0], edge_index[1]]

        # gat_conv w_ij
        edge_h = torch.cat([x[edge_index[0, :], :], x[edge_index[1, :], :]],
                           dim=1).t()
        e_w = torch.sigmoid(self.wij_att.mm(edge_h).squeeze())  #
        values = e_w * covariance_vector
        # softmax
        N = x.size(0)
        sp_edge_h = torch.sparse_coo_tensor(edge_index, -F.leaky_relu(values),
                                            size=(N, N))  # values() = E
        
        if sp_edge_h.dtype == torch.float16:                          
            sp_edge_h_float32 = sp_edge_h.to(torch.float32)
            sp_edge_h_float32 = sparse.softmax(sp_edge_h_float32, dim=1)
            # sp_edge_h_float32 = self.sparse_dropout(sp_edge_h_float32)
            h_prime = torch.sparse.mm(sp_edge_h_float32, x.to(torch.float32))  # (NxN) * (NxF') = (NxF')
            h_prime = h_prime.to(torch.float16)
        else:
            sp_edge_h = sparse.softmax(sp_edge_h, dim=1)
            # sp_edge_h = self.sparse_dropout(sp_edge_h)
            h_prime = torch.sparse.mm(sp_edge_h, x)  # (NxN) * (NxF') = (NxF')

        num_messages = message_x.size(0)
        new_message_x = torch.cat((message_x, h_prime[-num_messages:, :]), dim=-1)

        return self.out(new_message_x).squeeze(-1)

    def forward(self, graph, pof_graph):
        graph, pof_graph = self.trans_dimensions(graph, pof_graph)
        init_message_x, init_user_x = graph['message'].x, graph['user'].x

        _, message_x, user_x = self.conv1(graph, pof_graph)
        message_x, user_x = F.elu(message_x), F.elu(user_x)

        graph['user'].x = user_x
        graph['message'].x = message_x
        graph.user_series_features = user_x[graph.user_series]
        pof_graph.x = torch.cat((user_x, message_x), dim=0)
        # conv2
        cif, message_x, user_x = self.conv2(graph, pof_graph)
        # return
        message_x, user_x = F.elu(message_x), F.elu(user_x)

        res = self.predict(pof_graph, message_x, user_x)
        loss = self.compute_loss(graph, cif)

        return res, loss
