import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

import torch
from torch._C import device
import torch.nn as nn
from utils import Decay
from module import Interact, TLSTMCell, Attention_type, Attention_node


def NodeProjection(g, cfg):
    if cfg['use_ntype']:
        emb_dim = cfg['emb_dim']
        node2embs = nn.ParameterDict()
        for ntype in g.ntypes:
            M = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
            nn.init.xavier_uniform_(M, gain=1.414)
            node2embs[ntype] = M
    else:
        emb_dim = cfg['emb_dim']
        node2embs = nn.ParameterDict()
        M = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        nn.init.xavier_uniform_(M, gain=1.414)
        for ntype in g.ntypes:
            node2embs[ntype] = M

    return node2embs


class RelationNN(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        self.g = g
        emb_dim = cfg['emb_dim']
        stdv = 1. / math.sqrt(emb_dim)
        self.relationNN_s = nn.ModuleList()
        self.relationNN_g = nn.ModuleList()
        if cfg['use_etype']:
            for _ in self.g.etypes:
                if cfg['layer'] == 1:
                    self.MLP_s = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim))
                    nn.init.xavier_uniform_(self.MLP_s[0].weight, gain=1.414)
                    self.MLP_s[0].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_s[2].weight, gain=1.414)
                    self.MLP_s[2].bias.data.uniform_(-stdv, stdv)

                    self.MLP_g = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim))
                    nn.init.xavier_uniform_(self.MLP_g[0].weight, gain=1.414)
                    self.MLP_g[0].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_g[2].weight, gain=1.414)
                    self.MLP_g[2].bias.data.uniform_(-stdv, stdv)

                elif cfg['layer'] == 2:
                    self.MLP_s = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim))
                    nn.init.xavier_uniform_(self.MLP_s[0].weight, gain=1.414)
                    self.MLP_s[0].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_s[2].weight, gain=1.414)
                    self.MLP_s[2].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_s[4].weight, gain=1.414)
                    self.MLP_s[4].bias.data.uniform_(-stdv, stdv)

                    self.MLP_g = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(emb_dim, emb_dim))
                    nn.init.xavier_uniform_(self.MLP_g[0].weight, gain=1.414)
                    self.MLP_g[0].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_g[2].weight, gain=1.414)
                    self.MLP_g[2].bias.data.uniform_(-stdv, stdv)
                    nn.init.xavier_uniform_(self.MLP_g[4].weight, gain=1.414)
                    self.MLP_g[4].bias.data.uniform_(-stdv, stdv)

                self.relationNN_s.append(self.MLP_s)
                self.relationNN_g.append(self.MLP_g)
        else:
            if cfg['layer'] == 1:
                self.MLP_s = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                           nn.ReLU(),
                                           nn.Linear(emb_dim, emb_dim))
                nn.init.xavier_uniform_(self.MLP_s[0].weight, gain=1.414)
                self.MLP_s[0].bias.data.uniform_(-stdv, stdv)
                nn.init.xavier_uniform_(self.MLP_s[2].weight, gain=1.414)
                self.MLP_s[2].bias.data.uniform_(-stdv, stdv)

                # self.MLP_g = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                #                            nn.ReLU(),
                #                            nn.Linear(emb_dim, emb_dim))
                # nn.init.xavier_uniform_(self.MLP_g[0].weight, gain=1.414)
                # self.MLP_g[0].bias.data.uniform_(-stdv, stdv)
                # nn.init.xavier_uniform_(self.MLP_g[2].weight, gain=1.414)
                # self.MLP_g[2].bias.data.uniform_(-stdv, stdv)

                for _ in self.g.etypes:
                    self.relationNN_s.append(self.MLP_s)
                    # self.relationNN_g.append(self.MLP_g)
                    self.relationNN_g.append(self.MLP_s)

    def forward(self, n_type, e_type, emb):
        if n_type == 'source':
            for i, etype in enumerate(self.g.etypes):
                if etype == e_type:
                    return self.relationNN_s[i](emb)
        elif n_type == 'target':
            for i, etype in enumerate(self.g.etypes):
                if etype == e_type:
                    return self.relationNN_g[i](emb)



class DyHetGNN(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_dim = cfg['emb_dim']
        self.act = getattr(nn, cfg['act'])()
        self.n_users = cfg['n_userID']
        self.n_objects = cfg['n_objectID']
        self.criterion = nn.BCELoss()
        self.linear = nn.Linear(self.emb_dim, 1)
        self.drop_out = nn.Dropout(p=cfg['drop'])
        self.sigmoid = nn.Sigmoid()
        self.linear_s = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_g = nn.Linear(self.emb_dim, self.emb_dim)
        self.p = nn.Linear(self.emb_dim, 1)
        self.q = nn.Linear(self.emb_dim, 1)

        # initialize
        self.users_cell_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.users_cell_emb.weight.requires_grad = False
        self.objects_cell_emb = nn.Embedding(self.n_objects, self.emb_dim)
        self.objects_cell_emb.weight.requires_grad = False
        self.users_hid_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.users_hid_emb.weight.requires_grad = False
        self.objects_hid_emb = nn.Embedding(self.n_objects, self.emb_dim)
        self.objects_hid_emb.weight.requires_grad = False

        self.projection2embs = NodeProjection(g, cfg)
        self.relationNN = RelationNN(g, cfg)

        # Update Module
        ## interact unit
        self.inter_u, self.inter_v = Interact(g, cfg), Interact(g, cfg)
        ## update unit
        self.decay = Decay(cfg, cfg['decay_w'], cfg['decay_method'])
        if cfg['use_time_interval']:
            self.update_u, self.update_v = TLSTMCell(
                cfg['inter_dim'], self.emb_dim), TLSTMCell(cfg['inter_dim'], self.emb_dim)
        else:
            self.update_u, self.update_v = nn.LSTMCell(
                cfg['inter_dim'], self.emb_dim), nn.LSTMCell(cfg['inter_dim'], self.emb_dim)

        # Propagate Module
        # dual-level attention
        if cfg['use_att']:
            self.attention_type = Attention_type(g, cfg)
            self.attention_node = Attention_node(g, cfg)

    def forward(self, interactions):
        # userID(MyID), userID(otherID), flag(1, 2), time, rating(1, 0), labels: 1, 0
        n_edges = interactions.shape[0]
        self.user_cell2rep_u, self.user_hid2rep_u = dict(), dict()
        self.user_cell2rep_v, self.user_hid2rep_v = dict(), dict()
        self.object_cell2rep_v, self.object_hid2rep_v = dict(), dict()
        self.out_u, self.out_v = [], []
        out_e, e_labels, e_ratings = [], [], []
        # print(interactions)
        for i in range(n_edges):
            self.int_u, self.int_v = int(interactions[i, 0]), int(interactions[i, 1])
            u, v = interactions[i, 0], interactions[i, 1]
            self.u, self.v = u, v
            self.is_trust = 0
            if interactions[i, 2].item() == 1:
                self.is_trust = 1
            if self.is_trust:
                # print('This edge belongs to trusts......')
                bool_u = torch.cat((self.hg.out_degrees(u, 'trusts'), self.hg.in_degrees(
                    u, 'trusts'), self.hg.out_degrees(u, 'rates')))
                bool_v = torch.cat((self.hg.out_degrees(v, 'trusts'), self.hg.in_degrees(
                    v, 'trusts'), self.hg.out_degrees(v, 'rates')))
                self.hg.add_edges(u, v, {
                    'ratings': torch.tensor([interactions[i, 4].item()]),
                    'labels': torch.tensor([interactions[i, 5].item()])}, etype='trusts')
                # out_e.append(self.hg.edge_ids(u, v, etype='trusts'))
                e = self.hg.edge_ids(u, v, etype='trusts')
                e_labels.append(self.hg.edges['trusts'].data['labels'][e])
                e_ratings.append(self.hg.edges['trusts'].data['ratings'][e])
                last_t_u = self.hg.nodes['userID'].data['time'][u].clone()
                last_t_v = self.hg.nodes['userID'].data['time'][v].clone()
                self.hg.nodes['userID'].data['time'][u] = interactions[i, 3]
                self.hg.nodes['userID'].data['time'][v] = interactions[i, 3]
                self.decayed_u = self.decay(self.hg.nodes['userID'].data['time'][u] - last_t_u)
                self.decayed_v = self.decay(self.hg.nodes['userID'].data['time'][v] - last_t_v)
            else:
                # print('This edge belongs to rates......')
                bool_u = torch.cat((self.hg.out_degrees(u, 'trusts'), self.hg.in_degrees(
                    u, 'trusts'), self.hg.out_degrees(u, 'rates')))
                bool_v = self.hg.in_degrees(v, 'rates')
                self.hg.add_edges(u, v, {
                    'ratings': torch.tensor([interactions[i, 4].item()]),
                    'labels': torch.tensor([interactions[i, 5].item()])}, etype='rates')
                # out_e.append(self.hg.edge_ids(u, v, etype='rates'))
                e = self.hg.edge_ids(u, v, etype='rates')
                e_labels.append(self.hg.edges['rates'].data['labels'][e])
                e_ratings.append(self.hg.edges['rates'].data['ratings'][e])
                last_t_u = self.hg.nodes['userID'].data['time'][u].clone()
                last_t_v = self.hg.nodes['objectID'].data['time'][v].clone()
                self.hg.nodes['userID'].data['time'][u] = interactions[i, 3]
                self.hg.nodes['objectID'].data['time'][v] = interactions[i, 3]
                # print('test.........')
                # print(self.hg.nodes['hotelID'].data['time'][u])
                # print(last_t_u)
                # quit()
                self.decayed_u = self.decay(self.hg.nodes['userID'].data['time'][u] - last_t_u)
                self.decayed_v = self.decay(self.hg.nodes['objectID'].data['time'][v] - last_t_v)


           

            if self.int_u in self.user_cell2rep_u:
                self.cell_emb_u = self.user_cell2rep_u[self.int_u]
                self.hid_emb_u = self.user_hid2rep_u[self.int_u]
                # print('11111')
                # print(self.cell_emb_u)
            elif len(bool_u):
                self.cell_emb_u = self.users_cell_emb(u)
                self.hid_emb_u = self.users_hid_emb(u)
                # print('22222')
                # print(self.cell_emb_u)
            else:
                self.cell_emb_u = torch.matmul(
                    self.users_cell_emb(u), self.projection2embs['userID'])
                self.hid_emb_u = torch.matmul(
                    self.users_hid_emb(u), self.projection2embs['userID'])
                # print('33333')
                # print(self.cell_emb_u)

            if self.is_trust:
                if self.int_v in self.user_cell2rep_v:
                    self.cell_emb_v = self.user_cell2rep_v[self.int_v]
                    self.hid_emb_v = self.user_hid2rep_v[self.int_v]
                elif len(bool_v):
                    self.cell_emb_v = self.users_cell_emb(v)
                    self.hid_emb_v = self.users_hid_emb(v)
                else:
                    self.cell_emb_v = torch.matmul(
                        self.users_cell_emb(v), self.projection2embs['userID'])
                    self.hid_emb_v = torch.matmul(
                        self.users_hid_emb(v), self.projection2embs['userID'])
            else:
                if self.int_v in self.object_cell2rep_v:
                    self.cell_emb_v = self.object_cell2rep_v[self.int_v]
                    self.hid_emb_v = self.object_hid2rep_v[self.int_v]
                elif len(bool_v):
                    self.cell_emb_v = self.objects_cell_emb(v)
                    self.hid_emb_v = self.objects_hid_emb(v)
                else:
                    self.cell_emb_v = torch.matmul(
                        self.objects_cell_emb(v), self.projection2embs['objectID'])
                    self.hid_emb_v = torch.matmul(
                        self.objects_hid_emb(v), self.projection2embs['objectID'])
                
            if self.is_trust:
                self.e_info_u = self.inter_u(
                    self.hid_emb_u.view(-1, self.emb_dim), self.relationNN('target', 'trusts', self.hid_emb_v.view(-1, self.emb_dim)))
                self.e_info_v = self.inter_v(
                    self.relationNN('source', 'trusts', self.hid_emb_u.view(-1, self.emb_dim)), self.hid_emb_v.view(-1, self.emb_dim))
            else:
                self.e_info_u = self.inter_u(
                    self.hid_emb_u.view(-1, self.emb_dim), self.relationNN('target', 'rates', self.hid_emb_v.view(-1, self.emb_dim)))
                self.e_info_v = self.inter_v(
                    self.relationNN('source', 'rates', self.hid_emb_u.view(-1, self.emb_dim)), self.hid_emb_v.view(-1, self.emb_dim))
            
            """
            利用LSTM单元更新 s,g 的cell、hidden、node表示，更新后的表示写入字典， 
            out_s,out_g 添加更新后的 node 表示
            """
            self.update()

            """
            更新了 u,v 一阶邻居的表示，并更新了词典
            """
            if self.cfg['use_propagation']:
                if self.is_trust:
                    self.propagation(u, self.user_cell2rep_u[self.int_u], 'source')
                    self.propagation(v, self.user_cell2rep_v[self.int_v], 'target')
                else:
                    self.propagation(u, self.user_cell2rep_u[self.int_u], 'source')
                    self.propagation(v, self.object_cell2rep_v[self.int_v], 'target')

        # out_cell_u_idx = list(self.user_cell2rep_u.keys())
        # out_hid_u_idx = list(self.user_hid2rep_u.keys())
        # out_cell_v_idx = list(self.user_cell2rep_v.keys())
        # out_hid_v_idx = list(self.user_hid2rep_v.keys())
        # object_cell_rep_idx = list(self.object_cell2rep_v.keys())
        # object_hid_rep_idx = list(self.object_hid2rep_v.keys())

        with torch.no_grad():
            if self.user_cell2rep_u.values():
                out_cell_u_idx = list(self.user_cell2rep_u.keys())
                out_cell_u = torch.cat(list(self.user_cell2rep_u.values())).view(-1, self.emb_dim)
                self.users_cell_emb.weight[out_cell_u_idx] = out_cell_u
            if self.user_hid2rep_u.values():
                out_hid_u_idx = list(self.user_hid2rep_u.keys())
                out_hid_u = torch.cat(list(self.user_hid2rep_u.values())).view(-1, self.emb_dim)
                self.users_hid_emb.weight[out_hid_u_idx] = out_hid_u
            if self.user_cell2rep_v.values():
                out_cell_v_idx = list(self.user_cell2rep_v.keys())
                out_cell_v = torch.cat(list(self.user_cell2rep_v.values())).view(-1, self.emb_dim)
                self.users_cell_emb.weight[out_cell_v_idx] = out_cell_v
            if self.user_hid2rep_v.values():
                out_hid_v_idx = list(self.user_hid2rep_v.keys())
                out_hid_v = torch.cat(list(self.user_hid2rep_v.values())).view(-1, self.emb_dim)
                self.users_hid_emb.weight[out_hid_v_idx] = out_hid_v
            if self.object_cell2rep_v.values():
                object_cell_rep_idx = list(self.object_cell2rep_v.keys())
                object_cell_rep = torch.cat(list(self.object_cell2rep_v.values())).view(-1, self.emb_dim)
                self.objects_cell_emb.weight[object_cell_rep_idx] = object_cell_rep
            if self.object_hid2rep_v.values():
                object_hid_rep_idx = list(self.object_hid2rep_v.keys())
                object_hid_rep = torch.cat(list(self.object_hid2rep_v.values())).view(-1, self.emb_dim)
                self.objects_hid_emb.weight[object_hid_rep_idx] = object_hid_rep

        out_u = self.drop_out(torch.cat(self.out_u).view(-1, self.emb_dim))
        out_v = self.drop_out(torch.cat(self.out_v).view(-1, self.emb_dim))
        # if self.is_trust:
        #     for e in out_e:
        #         e_labels.append(self.hg.edges['trusts'].data['labels'][e])
        # else:
        #     for e in out_e:
        #         e_labels.append(self.hg.edges['rates'].data['labels'][e])
        # with torch.no_grad():
        #     self.users_cell_emb.weight[out_cell_u_idx] = out_cell_u
        #     self.users_hid_emb.weight[out_hid_u_idx] = out_hid_u
        #     self.users_cell_emb.weight[out_cell_v_idx] = out_cell_v
        #     self.users_hid_emb.weight[out_hid_v_idx] = out_hid_v
        #     self.objects_cell_emb.weight[object_cell_rep_idx] = object_cell_rep
        #     self.objects_hid_emb.weight[object_hid_rep_idx] = object_hid_rep

        # print('user_cell:', self.user_cell2rep_u)
        # print('object_cell:', self.object_cell2rep_v)
        # print('u:', out_u)
        # print('v:', out_v)
        # print('e_labels:', e_labels)
        # quit()

        return out_u, out_v, e_labels, e_ratings

    def update(self):
        if self.cfg['use_time_interval']:
            updated_u_hid, updated_u_cell = self.update_u(
                self.e_info_u.view(-1, self.cfg['inter_dim']), self.decayed_u.view(-1, 1),
                (self.hid_emb_u.view(-1, self.emb_dim), self.cell_emb_u.view(-1, self.emb_dim)))
            updated_v_hid, updated_v_cell = self.update_v(
                self.e_info_v.view(-1, self.cfg['inter_dim']), self.decayed_v.view(-1, 1),
                (self.hid_emb_v.view(-1, self.emb_dim), self.cell_emb_v.view(-1, self.emb_dim)))
        else:
            updated_u_hid, updated_u_cell = self.update_u(
                self.e_info_u.view(-1, self.cfg['inter_dim']), (self.hid_emb_u.view(-1, self.emb_dim), self.cell_emb_u.view(-1, self.emb_dim)))
            updated_v_hid, updated_v_cell = self.update_v(
                self.e_info_v.view(-1, self.cfg['inter_dim']), (self.hid_emb_v.view(-1, self.emb_dim), self.cell_emb_v.view(-1, self.emb_dim)))
        
        self.user_cell2rep_u[self.int_u] = updated_u_cell
        self.user_hid2rep_u[self.int_u] = updated_u_hid
        if self.is_trust:
            self.user_cell2rep_v[self.int_v] = updated_v_cell
            self.user_hid2rep_v[self.int_v] = updated_v_hid
        else:
            self.object_cell2rep_v[self.int_v] = updated_v_cell
            self.object_hid2rep_v[self.int_v] = updated_v_hid

        self.out_u.append(updated_u_hid)
        self.out_v.append(updated_v_hid)


    def propagation(self, node, node_rep, node_type):
        """
        trusts_s_neis_u: trusts: the edge type, s_neis: the source neighbors of u
        """
        if node_type == 'source':
            t = self.hg.nodes['userID'].data['time'][node]
        else:
            if self.is_trust:
                t = self.hg.nodes['userID'].data['time'][node]
            else:
                t = self.hg.nodes['objectID'].data['time'][node]


        trusts_decayed_neis_interval = []
        rates_decayed_neis_interval = []
        if node_type == 'source':
            trusts_g_neis = self.hg.successors(node, etype='trusts')
            rates_g_neis = self.hg.successors(node, etype='rates')
            # trusts_s_neis = self.hg.predecessors(node, etype='trusts')
            if self.is_trust:
                trusts_g_neis = trusts_g_neis[trusts_g_neis != self.v]
            else:
                rates_g_neis = rates_g_neis[rates_g_neis != self.v]
            # trusts_neis = torch.cat((trusts_g_neis, trusts_s_neis))
            if len(trusts_g_neis):
                trusts_g_neis_embs = self.get_embs(trusts_g_neis, self.user_cell2rep_v, self.users_cell_emb)  # neis * emb_dim
                for i, g_nei in enumerate(trusts_g_neis):
                    trusts_g_nei_interval = (t - self.hg.nodes['userID'].data['time'][g_nei])
                    decayed_trusts_g_nei_interval = self.decay(trusts_g_nei_interval)
                    trusts_decayed_neis_interval.append(decayed_trusts_g_nei_interval)
                trusts_decayed_neis_interval = torch.stack(trusts_decayed_neis_interval, dim=0)  # neis * 1
                if self.cfg['use_time_interval']:
                    trusts_g_neis_info = self.relationNN(
                        'source', 'trusts', node_rep).repeat(len(trusts_g_neis), 1) * trusts_decayed_neis_interval       # neis * emb_dim
                else:
                    trusts_g_neis_info = self.relationNN('source', 'trusts', node_rep)
                    

            # if len(trusts_s_neis):
            #     trusts_s_neis_embs = self.get_embs(trusts_s_neis, self.user_cell2rep_u, self.users_cell_emb)  # neis * emb_dim
            #     for i, s_nei in enumerate(trusts_s_neis):
            #         trusts_s_nei_interval = (t - self.hg.nodes['userID'].data['time'][s_nei])
            #         decayed_trusts_s_nei_interval = self.decay(trusts_s_nei_interval)
            #         trusts_decayed_neis_interval.append(decayed_trusts_s_nei_interval)
            #     trusts_decayed_neis_interval = torch.stack(trusts_decayed_neis_interval, dim=0)  # neis * 1
            #     if self.cfg['use_time_interval']:
            #         trusts_s_neis_info = self.relationNN(
            #             'target', 'trusts', node_rep).repeat(len(trusts_s_neis), 1) * trusts_decayed_neis_interval       # neis * emb_dim
            #     else:
            #         trusts_s_neis_info = self.relationNN('target', 'trusts', node_rep).repeat(len(trusts_s_neis), 1)

            if len(rates_g_neis):
                rates_g_neis_embs = self.get_embs(rates_g_neis, self.object_cell2rep_v, self.objects_cell_emb)  # neis * emb_dim
                for i, g_nei in enumerate(rates_g_neis):
                    rates_g_nei_interval = (t - self.hg.nodes['objectID'].data['time'][g_nei])
                    decayed_rates_g_nei_interval = self.decay(rates_g_nei_interval)
                    rates_decayed_neis_interval.append(decayed_rates_g_nei_interval)
                rates_decayed_neis_interval = torch.stack(rates_decayed_neis_interval, dim=0)  # neis * 1
                if self.cfg['use_time_interval']:
                    rates_g_neis_info = self.relationNN(
                        'source', 'rates', node_rep).repeat(len(rates_g_neis), 1) * rates_decayed_neis_interval       # neis * emb_dim
                else:
                    rates_g_neis_info = self.relationNN('source', 'rates', node_rep)
                        
            if self.cfg['use_att']:   
                if len(trusts_g_neis) >0 and len(rates_g_neis) > 0:
                    trusts_g_neis_info, rates_g_neis_info = self.attention_type(node_rep, trusts_g_neis_info, rates_g_neis_info)
                    trusts_g_neis_cell = trusts_g_neis_embs + self.linear_s(trusts_g_neis_info)
                    rates_g_neis_cell = rates_g_neis_embs + self.linear_s(rates_g_neis_info)
                    for i, g_nei in enumerate(trusts_g_neis):
                        self.user_cell2rep_v[int(g_nei)] = trusts_g_neis_cell[i].view(-1, self.emb_dim)
                        self.user_hid2rep_v[int(g_nei)] = self.act(trusts_g_neis_cell[i]).view(-1, self.emb_dim)
                    for i, g_nei in enumerate(rates_g_neis):
                        self.object_cell2rep_v[int(g_nei)]=rates_g_neis_cell[i].view(-1, self.emb_dim)
                        self.object_hid2rep_v[int(g_nei)]=self.act(rates_g_neis_cell[i]).view(-1, self.emb_dim)
                elif len(trusts_g_neis) > 0 and len(rates_g_neis) ==0:
                    trusts_g_neis_info = self.attention_node(node_rep, trusts_g_neis_info)
                    trusts_g_neis_cell = trusts_g_neis_embs + self.linear_s(trusts_g_neis_info)
                    for i, g_nei in enumerate(trusts_g_neis):
                        self.user_cell2rep_v[int(g_nei)] = trusts_g_neis_cell[i].view(-1, self.emb_dim)
                        self.user_hid2rep_v[int(g_nei)] = self.act(trusts_g_neis_cell[i]).view(-1, self.emb_dim)
                elif len(trusts_g_neis) == 0 and len(rates_g_neis) > 0:
                    rates_g_neis_info = self.attention_node(node_rep, rates_g_neis_info)
                    rates_g_neis_cell = rates_g_neis_embs + self.linear_s(rates_g_neis_info)
                    for i, g_nei in enumerate(rates_g_neis):
                        self.object_cell2rep_v[int(g_nei)] = rates_g_neis_cell[i].view(-1, self.emb_dim)
                        self.object_hid2rep_v[int(g_nei)] = self.act(rates_g_neis_cell[i]).view(-1, self.emb_dim)
        else:
            if self.is_trust:
                trusts_g_neis = self.hg.successors(node, etype='trusts')
                rates_g_neis = self.hg.successors(node, etype='rates')
                # trusts_s_neis = trusts_s_neis[trusts_s_neis != self.u]

                if len(trusts_g_neis):
                    trusts_g_neis_embs = self.get_embs(trusts_g_neis, self.user_cell2rep_v, self.users_cell_emb)  # neis * emb_dim
                    for i, g_nei in enumerate(trusts_g_neis):
                        trusts_g_nei_interval = (t - self.hg.nodes['userID'].data['time'][g_nei])
                        decayed_trusts_g_nei_interval = self.decay(trusts_g_nei_interval)
                        trusts_decayed_neis_interval.append(decayed_trusts_g_nei_interval)
                    trusts_decayed_neis_interval = torch.stack(trusts_decayed_neis_interval, dim=0)  # neis * 1
                    if self.cfg['use_time_interval']:
                        trusts_g_neis_info = self.relationNN(
                            'source', 'trusts', node_rep).repeat(len(trusts_g_neis), 1) * trusts_decayed_neis_interval       # neis * emb_dim
                    else:
                        trusts_g_neis_info = self.relationNN('source', 'trusts', trusts_g_neis_embs)

                if len(rates_g_neis):
                    rates_g_neis_embs = self.get_embs(rates_g_neis, self.object_cell2rep_v, self.objects_cell_emb)  # neis * emb_dim
                    for i, g_nei in enumerate(rates_g_neis):
                        rates_g_nei_interval = (t - self.hg.nodes['objectID'].data['time'][g_nei])
                        decayed_rates_g_nei_interval = self.decay(rates_g_nei_interval)
                        rates_decayed_neis_interval.append(decayed_rates_g_nei_interval)
                    rates_decayed_neis_interval = torch.stack(rates_decayed_neis_interval, dim=0)  # neis * 1
                    if self.cfg['use_time_interval']:
                        rates_g_neis_info = self.relationNN(
                            'source', 'rates', node_rep).repeat(len(rates_g_neis), 1) * rates_decayed_neis_interval       # neis * emb_dim
                    else:
                        rates_g_neis_info = self.relationNN('target', 'rates', rates_g_neis_embs)
                
                if self.cfg['use_att']:
                    if len(trusts_g_neis) > 0 and len(rates_g_neis) > 0:
                        trusts_g_neis_info, rates_g_neis_info = self.attention_type(node_rep, trusts_g_neis_info, rates_g_neis_info)
                        trusts_g_neis_cell = trusts_g_neis_embs + self.linear_s(trusts_g_neis_info)
                        rates_g_neis_cell = rates_g_neis_embs + self.linear_s(rates_g_neis_info)
                        for i, g_nei in enumerate(trusts_g_neis):
                            self.user_cell2rep_v[int(g_nei)] = trusts_g_neis_cell[i].view(-1, self.emb_dim)
                            self.user_hid2rep_v[int(g_nei)] = self.act(trusts_g_neis_cell[i]).view(-1, self.emb_dim)
                        for i, g_nei in enumerate(rates_g_neis):
                            self.object_cell2rep_v[int(g_nei)] = rates_g_neis_cell[i].view(-1, self.emb_dim)
                            self.object_hid2rep_v[int(g_nei)] = self.act(rates_g_neis_cell[i]).view(-1, self.emb_dim)
                    elif len(trusts_g_neis) > 0 and len(rates_g_neis) == 0:
                        trusts_g_neis_info = self.attention_node(node_rep, trusts_g_neis_info)
                        trusts_g_neis_cell = trusts_g_neis_embs + self.linear_s(trusts_g_neis_info)
                        for i, g_nei in enumerate(trusts_g_neis):
                            self.user_cell2rep_v[int(g_nei)] = trusts_g_neis_cell[i].view(-1, self.emb_dim)
                            self.user_hid2rep_v[int(g_nei)] = self.act(trusts_g_neis_cell[i]).view(-1, self.emb_dim)
                    elif len(trusts_g_neis) == 0 and len(rates_g_neis) > 0:
                        rates_g_neis_info = self.attention_node(node_rep, rates_g_neis_info)
                        rates_g_neis_cell = rates_g_neis_embs + self.linear_s(rates_g_neis_info)
                        for i, g_nei in enumerate(rates_g_neis):
                            self.object_cell2rep_v[int(g_nei)] = rates_g_neis_cell[i].view(-1, self.emb_dim)
                            self.object_hid2rep_v[int(g_nei)] = self.act(rates_g_neis_cell[i]).view(-1, self.emb_dim)
            else:
                rates_s_neis = self.hg.predecessors(node, etype='rates')
                rates_s_neis = rates_s_neis[rates_s_neis != self.u]

                if len(rates_s_neis):
                    rates_s_neis_embs = self.get_embs(rates_s_neis, self.user_cell2rep_u, self.users_cell_emb)  # neis * emb_dim
                    for i, s_nei in enumerate(rates_s_neis):
                        rates_s_nei_interval = (t - self.hg.nodes['userID'].data['time'][s_nei])
                        decayed_rates_s_nei_interval = self.decay(rates_s_nei_interval)
                        rates_decayed_neis_interval.append(decayed_rates_s_nei_interval)
                    rates_decayed_neis_interval = torch.stack(rates_decayed_neis_interval, dim=0)  # neis * 1
                    if self.cfg['use_time_interval']:
                        rates_s_neis_info = self.relationNN(
                            'target', 'rates', node_rep).repeat(len(rates_s_neis), 1) * rates_decayed_neis_interval       # neis * emb_dim
                    else:
                        rates_s_neis_info = self.relationNN('target', 'rates', node_rep).repeat(len(rates_s_neis), 1)
                    
                    if self.cfg['use_att']:
                        rates_s_neis_info = self.attention_node(node_rep, rates_s_neis_info)
                        rates_s_neis_cell = rates_s_neis_embs + self.linear_g(rates_s_neis_info)
                        for i, s_nei in enumerate(rates_s_neis):
                            self.user_cell2rep_u[int(s_nei)] = rates_s_neis_cell[i].view(-1, self.emb_dim)
                            self.user_hid2rep_u[int(s_nei)] = self.act(rates_s_neis_cell[i]).view(-1, self.emb_dim)


            
    def get_embs(self, neis: torch.tensor, rep: dict, embs: nn.Embedding):
        ret_embs = embs(neis)
        for nei in neis:
            if int(nei) in rep:
                ret_embs[torch.nonzero(nei == neis)[0]] = rep[int(nei)]
        return ret_embs


    def loss(self, hg, interactions):
        self.hg = hg
        out_u_embs, out_v_embs, labels, ratings = self.forward(interactions)
        labels = torch.stack(labels, dim=0)
        # print('out_u', out_u_embs)
        # print('out_v', out_v_embs)
        out_edge_embs = out_u_embs * out_v_embs

        # out_edge_embs = self.p.weight * out_u_embs + self.q.weight * out_v_embs

        # ratings = torch.unsqueeze(
        #     torch.cat(ratings, 0), 1).to(self.cfg['device'])

        # out_edge_embs = ratings * \
        #     self.act(self.p.weight * out_u_embs + self.q.weight * out_v_embs)

        preds = self.sigmoid(self.linear(out_edge_embs))
        eps = torch.FloatTensor([10 ** -10])
        loss = self.criterion(preds.cpu() + eps, labels.float())

        #accuracy
        labels_preds = preds > 1 - preds
        acc = float((labels_preds.cpu() == labels).sum().item()) / labels.shape[0]

        # auc
        # auc = roc_auc_score(labels.numpy(), preds.cpu().detach().numpy())
        return loss, acc, labels.numpy(), preds.cpu().detach().numpy()
