import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import pdb

class Interact(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear_u = nn.Linear(cfg['emb_dim'], cfg['inter_dim'])
        self.linear_v = nn.Linear(cfg['emb_dim'], cfg['inter_dim'])
        self.act = getattr(nn, cfg['act'])()

    def forward(self, emb_u: nn.Embedding, emb_v: nn.Embedding):
        return self.act(self.linear_u(emb_u) + self.linear_v(emb_v))


class TLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # cell to short memory Eq(2)
        self.c2s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        # combine 4 gates' computation together
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size)

    def forward(self, input, decayed_delta_t, states: tuple):
        hidden, cell = states
        cell_short = self.c2s(cell)  # Eq(2)
        cell_new = cell - cell_short + cell_short * decayed_delta_t  # Eq(5)

        gates = self.i2h(input) + self.h2h(hidden)
        # print('test....')
        # print(input)
        # print(hidden)
        # quit()
        ingate, forgate, cellgate, outgate = gates.chunk(4, 1)  # 4块，dim = 1
        ingate = self.sigmoid(ingate)
        forgate = self.sigmoid(forgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cell_output = forgate * cell_new + ingate * cellgate
        hidden_output = outgate * self.tanh(cell_output)
        return hidden_output, cell_output


# g.predecessors,  g.successors
class Attention_type(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        self.act = cfg['act']
        self.softmax = nn.Softmax(dim=0)
        self.type_att = nn.Parameter(torch.ones(
            cfg['emb_dim'] * 2, 1), requires_grad=True)
        self.node_att = nn.Parameter(torch.ones(
            cfg['emb_dim'] * 2, 1), requires_grad=True)
        self.att = nn.Sequential()

    def forward(self, node_emb, trusts_neis_embs, rates_neis_embs):
        
        # Type Attention
        trusts_rep = trusts_neis_embs.sum(0)
        # writes_rep = writes_neis_embs.sum(0)
        rates_rep = rates_neis_embs.sum(0)
        type_cat_embs = torch.stack((trusts_rep, rates_rep), 0)
        type_node_embs = node_emb.repeat(2, 1)
        node_cat_type = torch.cat((type_node_embs, type_cat_embs), dim=1)
        A = torch.matmul(node_cat_type, self.type_att)  # 2 * 1
        A = self.softmax(A)  # 2 * 1

        # Node Attention
        # order: trusts -> writes -> rates
        # node_emb: 1 * emb_dim
        trusts_node_embs = node_emb.repeat(len(trusts_neis_embs), 1)   # neis * emb_dim
        # neis * (2 * emb_dim )
        trusts_node_cat_nei = torch.cat((trusts_node_embs, trusts_neis_embs), dim=1)
        trusts_B = torch.matmul(trusts_node_cat_nei, self.node_att)  # neis * 1
        trusts_B = self.softmax(trusts_B)
        ret_trusts_neis_embs = A[0] * trusts_B * trusts_neis_embs  # neis * emb_dim

        # writes_node_embs = node_emb.repeat(len(writes_neis_embs), 1)   # neis * emb_dim
        # # neis * (2 * emb_dim )
        # writes_node_cat_nei = torch.cat(
        #     (writes_node_embs, writes_neis_embs), dim=1)
        # writes_B = torch.matmul(writes_node_cat_nei, self.node_att)  # neis * 1
        # writes_B = self.softmax(writes_B)
        # ret_writes_neis_embs = A[1] * writes_B * writes_neis_embs  # neis * emb_dim

        rates_node_embs = node_emb.repeat(len(rates_neis_embs), 1)   # neis * emb_dim
        # neis * (2 * emb_dim )
        rates_node_cat_nei = torch.cat(
            (rates_node_embs, rates_neis_embs), dim=1)
        rates_B = torch.matmul(rates_node_cat_nei, self.node_att)  # neis * 1
        rates_B = self.softmax(rates_B)
        ret_rates_neis_embs = A[1] * rates_B * rates_neis_embs  # neis * emb_dim
        return ret_trusts_neis_embs,  ret_rates_neis_embs


class Attention_node(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        self.act = cfg['act']
        self.softmax = nn.Softmax(dim=0)
        self.type_att = nn.Parameter(torch.ones(
            cfg['emb_dim'] * 2, 1), requires_grad=True)
        self.node_att = nn.Parameter(torch.ones(
            cfg['emb_dim'] * 2, 1), requires_grad=True)
        self.att = nn.Sequential()

    def forward(self, node_emb, neis_embs):
        # Type Attention
        # relation2neis = dict()
        # for etype in g.etypes:
        #     neis = torch.cat(g.predecessors(u_id, etype), g.successors(u_id, etype))
        #     relation2neis[etype]=neis

        # node_emb: 1 * emb_dim
        # pdb.set_trace()
        assert len(neis_embs) > 0
        # neis_embs = node_emb.repeat(0, 1)
        # assert neis_embs.shape[0] == 0
        node_embs = node_emb.repeat(len(neis_embs), 1)   # neis * emb_dim
        # (neis, 2*emb_dim)
        node_cat_nei = torch.cat((node_embs, neis_embs), dim=1)
        B = torch.matmul(node_cat_nei, self.node_att)  # neis * 1
        B = self.softmax(B)
        ret_neis_embs = B * neis_embs  # neis * emb_dim
        return ret_neis_embs
