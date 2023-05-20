import torch
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch.multiprocessing import Queue
from torch.nn import Linear
import dgl
import pickle
import os
import dgl.function as fn
from dgl.base import DGLError
from functools import partial
from collections import Counter
from tqdm import tqdm


class CSIP(nn.Module):
    def __init__(self, in_feats, out_feats, dist_dim, num_neighbor, scopes, dist_feat_in, fd, act=None,
                 ho1=False, elem_gate=False, merge='sum'):
        super(CSIP, self).__init__()
        self.merge = merge
        self.tl = ['morning', 'midday', 'night', 'late-night']
        self.agg = SpatialGraphConv({
            '00': CSIP_Hop(in_feats, out_feats, dist_dim, scopes, dist_feat_in, fd, act, ho1, elem_gate),
            '01': CSIP_Hop(in_feats, out_feats, dist_dim, scopes, dist_feat_in, fd, act, ho1, elem_gate),
            '10': CSIP_Hop(in_feats, out_feats, dist_dim, scopes, dist_feat_in, fd, act, ho1, elem_gate),
            '11': CSIP_Hop(in_feats, out_feats, dist_dim, scopes, dist_feat_in, fd, act, ho1, elem_gate)},
            aggregate='sum')
        self.epp = EPP(out_feats, out_feats, dist_dim, scopes, dist_feat_in, num_neighbor, 0., 0.,
                       t=True, acti=act)

    def forward(self, graph, feat):
        graph = graph.local_var()
        h_list = []
        for key in self.tl:
            etype_list = [key + ids for ids in ['00', '01', '10', '11']]
            subg = graph.edge_type_subgraph(etype_list)
            h = self.agg(subg, {'p': feat})
            h_list.append(h['p'])

        h_t_list = []
        for i, key in enumerate(self.tl):
            i1 = (i - 1) % 4
            i2 = (i + 1) % 4
            etype_list = [e for e in graph.etypes if key == e]
            etype_list += [e for e in graph.etypes if self.tl[i1] == e]
            etype_list += [e for e in graph.etypes if self.tl[i2] == e]
            h_t = torch.stack([h_list[i1], h_list[i], h_list[i2]], dim=1)
            if self.merge == 'sum':
                h_t = torch.sum(h_t, dim=1)
            if self.merge == 'mean':
                h_t = torch.mean(h_t, dim=1)
            if self.merge == 'max':
                h_t = torch.max(h_t, dim=1)[0]

            subg = dgl.to_homogeneous(graph.edge_type_subgraph(etype_list), ndata=['loc'])
            h_t = self.epp(subg, h_t)
            h_t_list.append(h_t)
        return h_t_list


class CSIP_Hop(nn.Module):
    def __init__(self, in_feats, out_feats, dist_dim, boundaries, dist_feature_inputed, feat_drop, activation=None,
                 hop1_fc=False,
                 elem_gate=False):
        super(CSIP_Hop, self).__init__()
        self.jump = hop1_fc
        self.gatting = elem_gate
        self.scopes = boundaries
        self.fin = dist_feature_inputed
        self.fc2 = nn.Linear(in_feats, out_feats, bias=False)
        self.fcd = nn.Linear(dist_dim, out_feats, bias=False)
        if self.jump:
            self.fc1 = nn.Linear(in_feats, out_feats, bias=False)

        self.fc_w1 = nn.Linear(2 * out_feats, out_feats, bias=False)
        self.fc_w2 = nn.Linear(2 * out_feats, out_feats, bias=False)
        if not self.gatting:
            self.vec_a = nn.Linear(out_feats, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.fdout = nn.Dropout(feat_drop)
        self.act = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self.jump:
            nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fcd.weight)
        nn.init.xavier_uniform_(self.fc_w1.weight)
        nn.init.xavier_uniform_(self.fc_w2.weight)
        if not self.gatting:
            nn.init.xavier_uniform_(self.vec_a.weight)

    def cal_dist(self, edges):
        dist = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        fin = self.fcd(self.fin(torch.bucketize(dist, self.scopes)))
        return {'dist': fin}

    def passing_message(self, edges):
        w1 = torch.cat([edges.dst['h2'], edges.data['dist']], dim=1)
        w2 = torch.cat([edges.data['h1'], edges.src['h2']], dim=1)
        w1 = self.fc_w1(w1)
        w2 = self.fc_w2(w2)
        scores = w1 + w2
        if not self.gatting:
            scores = self.vec_a(scores)
        beta = self.sig(scores)
        h = beta * edges.src['h2'] + edges.data['h1']
        h = edges.dst['d0'] * edges.src['d0'] * h
        return {'he': h}

    def forward(self, graph, feat):
        graph = graph.local_var()
        graph.apply_edges(self.cal_dist)
        feat = self.fdout(feat)
        h2 = self.fc2(feat)
        if self.jump:
            h1 = self.fc1(feat)
        else:
            h1 = h2

        mid_ids = graph.edata['mid']
        graph.ndata['h2'] = h2
        graph.edata['h1'] = h1[mid_ids]
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d0'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.passing_message)

        graph.update_all(message_func=fn.copy_edge('he', 'm'), reduce_func=fn.sum(msg='m', out='x'))
        feat = graph.nodes['p'].data.pop('x')
        if self.act:
            feat = self.act(feat)
        return feat


class EPP(nn.Module):
    def __init__(self,
                 f,
                 o,
                 d,
                 b,
                 de,
                 ne=5,
                 fout=0.,
                 aout=0., ri=False, t=False, acti=None):
        super(EPP, self).__init__()
        self.num = ne
        self.out_feats = o
        self.dist_dim = d
        self.transform = t
        if self.transform:
            self.fc = nn.Linear(f, o, bias=False)
            attn_in_feats = 2 * o
        else:
            attn_in_feats = 2 * f
        self.agg_fc = nn.Linear(o * 2, o, bias=True)
        self.boundaries = b
        self.feature_inputed = de
        self.G = nn.Linear(d, o, bias=False)

        self.feat_drop = nn.Dropout(fout)
        self.attn_drop = nn.Dropout(aout)
        self.tanh = nn.Tanh()
        if ri:
            if f != o:
                self.res_fc = nn.Linear(
                    f, o, bias=False)
            else:
                self.res_fc = Identity_NN()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = acti

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.agg_fc.weight)
        nn.init.xavier_normal_(self.G.weight)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))

    def combine_dist(self, edges):
        dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
        dist_feature_inputed = self.G(self.feature_inputed(torch.bucketize(dist, self.boundaries)))
        h_dist = self.combine(dist_feature_inputed * edges.src['h'])
        return {'dh': h_dist}

    def combine_double_dist(self, edges):
        dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        dist_ = (edges.src['loc'].unsqueeze(1).repeat(1, self.num, 1) - edges.data['inter_pos']).norm(p=2, dim=-1)
        dist_feature_inputed1 = self.G(self.feature_inputed(torch.bucketize(dist1, self.boundaries)))
        dist_feature_inputed_ = self.G(
            self.feature_inputed(torch.bucketize(dist_, self.boundaries)).view(-1, self.dist_dim))

        dist_feature_inputed_dot_h = (
                dist_feature_inputed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']).mean(dim=1)
        h_dist = edges.dst['d0'] * edges.src['d0'] * self.agg_fc(
            torch.cat([dist_feature_inputed1 * edges.src['h'], dist_feature_inputed_dot_h], dim=1))
        return {'h': h_dist}

    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num, replace=True)
        src_new = sg.edges()[0].view(-1, num)
        return src_new.to(cuda)

    def message_func(self, edges):
        return {'h': edges.data['h'], 't': edges.data['_TYPE']}

    def forward(self, graph, feat):
        graph = graph.local_var()
        if self.transform:
            feat = self.fc(feat)
        graph.ndata['h'] = feat
        locations = graph.ndata['loc']

        inter_ids = self.building_inds_random(graph, num=self.num)
        graph.edata['inter_pos'] = locations[inter_ids]
        graph.edata['inter_h'] = feat[inter_ids]
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d0'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.combine_double_dist)
        graph.update_all(fn.copy_e('h', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # acti
        if self.activation:
            rst = self.activation(rst)
        return rst


class TimeDiscriminator(nn.Module):
    def __init__(self, n_h):
        super(TimeDiscriminator, self).__init__()
        self.f_i = nn.Linear(n_h, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, fem, fin, grid_sizes, pos_samples, neg_samples, pos_bias=None,
                neg_bias=None):
        fin = self.f_i(fin)
        pos_fin = fin[pos_samples]
        grid_fin = dgl.ops.segment_reduce(grid_sizes, pos_fin, 'mean')

        fem = self.f_i(fem)
        pos_fin = fem[pos_samples]
        neg_emebd = fem[neg_samples]

        pos_grid_feature_inputed_list = []
        for i, k in enumerate(grid_sizes):
            pos_grid_feature_inputed_list += [grid_fin[i].repeat(k, 1)]
        pos_grid_feature_inputed = th.cat(pos_grid_feature_inputed_list)

        grid_sizes_neg = grid_sizes * int(neg_samples.shape[0] / pos_samples.shape[0])
        neg_grid_feature_inputed_list = []
        for i, k in enumerate(grid_sizes_neg):
            neg_grid_feature_inputed_list += [grid_fin[i].repeat(k, 1)]
        neg_grid_feature_inputed = th.cat(neg_grid_feature_inputed_list)

        pos_logits = th.squeeze(self.f_k(pos_fin, pos_grid_feature_inputed), 1)
        neg_logits = th.squeeze(self.f_k(neg_emebd, neg_grid_feature_inputed), 1)

        if pos_bias is not None:
            pos_logits += pos_bias
        if neg_bias is not None:
            neg_logits += neg_bias
        logits = th.cat((pos_logits, neg_logits))
        return logits


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation, bias=True):
        super(MLPClassifier, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc_h = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc_o = nn.Linear(hidden_dim, 1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_h.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        if self.bias:
            self.fc_h.bias.data.zero_()
            self.fc_o.bias.data.zero_()

    def forward(self, input_feat):
        h = self.activation(self.fc_h(input_feat))
        return self.fc_o(h)


class EGNN(nn.Module):
    def __init__(self, i1, h1, rel_number, neig, dout=0, rp1=0, d1_i=0, scope_i=None):
        super(EGNN, self).__init__()
        dist_dim = h1
        self.d0 = rp1
        self.d1 = d1_i
        self.rel = nn.Parameter(torch.Tensor(rel_number, h1))
        self.scope = scope_i
        self.feature_inputedding = torch.nn.feat_in(len(self.scope) + 1, dist_dim)

        self.b = TimeDiscriminator(h1)
        self.feature_inputedding = torch.nn.feat_in(i1, h1)
        self.nn = CSIP(h1, h1, dist_dim, neig, self.scope, self.feature_inputedding, dout, act=F.relu, ho1=False,
                       elem_gate=False, merge='sum')
        self.mlp = nn.ModuleList()
        for _ in range(4):
            self.mlp.append(MLPClassifier(h1 * 2, h1, activation=F.relu))
        nn.init.xavier_uniform_(self.rel, gain=nn.init.calculate_gain('relu'))

    def calc_evo_score(self, e1, e2, pairs, idx):
        r_emebed = e1[pairs[:, 0]] * e1[pairs[:, 1]]
        r2_feature_inputed = e2[pairs[:, 0]] * e2[pairs[:, 1]]
        score = self.mlp[idx](torch.cat([r_emebed, r2_feature_inputed], dim=-1))
        return score

    def forward(self, g, h):
        h = self.feature_inputedding(h.squeeze())
        h = self.nn.forward(g, h)
        return h

    def get_loss(self, g, feature_inputed, evo_pairs, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels):
        predict_loss = 0
        for idx in range(4):
            evo_score = self.calc_evo_score(feature_inputed[idx], feature_inputed[(idx + 1) % 4], evo_pairs[idx], idx)
            logits = self.b(feature_inputed[idx], feature_inputed[(idx + 1) % 4], grid_sizes, pos_samples, neg_samples)
            predict_loss += self.d1 * F.binary_cross_entropy_with_logits(logits, grid_labels)
            predict_loss += self.d0 * F.binary_cross_entropy_with_logits(evo_score, evo_labels[idx])
        w_dist = self.feature_inputedding.weight
        predict_loss += 0.000001 * (torch.sum((w_dist[1:, :] - w_dist[:-1, :]) ** 2))
        return predict_loss


class EGNNPred(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, dropout=0, boundaries=None):
        super(EGNNPred, self).__init__()
        dist_dim = h_dim
        self.w_rel = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.boundaries = boundaries
        self.dist_feat_in = torch.nn.feat_in(len(self.boundaries) + 1, dist_dim)

        self.feat_in = torch.nn.feat_in(in_dim, h_dim)
        self.gnn = CSIP(h_dim, h_dim, dist_dim, num_neighbor, self.boundaries, self.dist_feat_in, dropout,
                        act=F.relu, ho1=False, elem_gate=False, merge='sum')
        nn.init.xavier_uniform_(self.w_rel, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, feat_in, triplets):
        s = feat_in[triplets[:, 0]]
        r = self.w_rel[triplets[:, 1]]
        o = feat_in[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def select_o(self, triplets_to_filter, target_s, target_r, target_o, train_ids):
        target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
        filtered_o = []
        if (target_s, target_r, target_o) in triplets_to_filter:
            triplets_to_filter.remove((target_s, target_r, target_o))
        for o in train_ids:
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered_o.append(o)
        return torch.LongTensor(filtered_o).cuda()

    def select_rs(self, feature_emb, test_triplets, train_triplets, valid_triplets):
        with torch.no_grad():
            s = test_triplets[:, 0]
            r = test_triplets[:, 1]
            o = test_triplets[:, 2]
            test_size = test_triplets.shape[0]
            triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
            train_ids = torch.unique(train_triplets[:, [0, 2]]).tolist()
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
            num_entities = feature_emb.shape[0]
            ranks = []

            for idx in range(test_size):
                target_s = s[idx]
                target_r = r[idx]
                target_o = o[idx]

                filtered_o = self.select_o(triplets_to_filter, target_s, target_r, target_o, train_ids)
                if len((filtered_o == target_o).nonzero()) == 0:
                    continue
                target_o_idx = int((filtered_o == target_o).nonzero())
                fis = feature_emb[target_s]
                fir = self.w_rel[target_r]
                fio = feature_emb[filtered_o]
                feature_input_triplet = fis * fir * fio
                scores = torch.sig(torch.sum(feature_input_triplet, dim=1))
                _, indices = torch.sort(scores, descending=True)
                rank = int((indices == target_o_idx).nonzero())
                ranks.append(rank)

        return np.array(ranks)

    def nn_for(self, g, h):
        h = self.feat_in(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, feature_inputed, triplets, labels):
        loss = 0
        for idx in range(4):
            score = self.calc_score(feature_inputed[idx], triplets[idx])
            loss += F.binary_cross_entropy_with_logits(score, labels[idx])
        return loss


def dynamic_dist(bin_num, coords, train):
    src_coord = coords[train[:, 0]]
    dst_coord = coords[train[:, 2]]
    delat = src_coord - dst_coord
    all_dist = np.sqrt((delat * delat).sum(1))

    all_dist = sorted(all_dist)
    cnt = len(all_dist)
    res = []
    for i in range(0, cnt, int(cnt / bin_num)):
        res += [all_dist[i]]
    return torch.tensor(res[1:]).cuda()


def select_time_data(data, rel_dict, key='morning'):
    result = []
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in rel_dict.items() if key in k]
        data_selected = [data[data[:, 1] == i] for i in ids]
        result.append(torch.cat(data_selected))
    return result


# Utility function for building training and testing graphs
def generate_batch_grids(grid_dict, grid_batch_size, negative_ratio=2.0, sampling_hop=2):
    all_grid = np.arange(len(grid_dict))
    grid_batch_size = min(len(all_grid), grid_batch_size)
    inds = np.random.choice(all_grid, grid_batch_size)
    keys = list(grid_dict.keys())
    batch_keys = [keys[i] for i in inds]
    pos_samples = np.concatenate([grid_dict[k] for k in batch_keys])
    neg_samples = negative_sampling_grid_near(batch_keys, grid_dict, negative_ratio, negative_ratio)
    grid_sizes = [len(grid_dict[k]) for k in batch_keys]
    labels = np.zeros(len(pos_samples) * (negative_ratio + 1), dtype=np.float32)
    labels[: len(pos_samples)] = 1
    return np.array(grid_sizes), np.array(pos_samples), np.array(neg_samples), np.array(labels)


def negative_sampling_grid_near(keys, grid_dict, sampling_hop, negative_ratio):
    all_nodes = set(np.concatenate([ll for ll in grid_dict.values()]))
    all_grids = list(grid_dict.keys())
    keys_expand = np.tile(np.expand_dims(keys, 1), (1, len(all_grids), 1))
    manha = np.abs(keys_expand - all_grids).sum(-1)
    inds_x, inds_y = np.where((manha > sampling_hop) & (manha < sampling_hop + 4))
    grids_expand = np.repeat(np.expand_dims(all_grids, 0), len(keys), axis=0)
    sampling_grids = grids_expand[inds_x, inds_y]

    neg_samples = []
    random_cnt = 0
    for i in range(len(keys)):
        neg_grids = sampling_grids[inds_x == i]
        if len(neg_grids) == 0:
            nodes_range = list(all_nodes - set(grid_dict[keys[i]]))
            random_cnt += 1
        else:
            nodes_range = np.concatenate([grid_dict[tuple(g)] for g in neg_grids])
        neg_nodes = np.random.choice(nodes_range, negative_ratio * len(grid_dict[keys[i]]))
        neg_samples.append(neg_nodes)
    return np.concatenate(neg_samples)


def build_dynamic_labels(data, rel_dict, num_nodes):
    pair_list, label_list = [[] for _ in range(4)], [[] for _ in range(4)]
    for rel in ['competitive', 'complementary']:
        g_list = [np.zeros((num_nodes, num_nodes)) for _ in range(4)]
        for i, key in enumerate(['_morning', '_midday', '_night', '_late-night']):
            ids = [v for k, v in rel_dict.items() if key in k and rel in k]
            data_selected = [data[data[:, 1] == i] for i in ids]
            data_selected = np.concatenate(data_selected)
            g_list[i][data_selected[:, 0], data_selected[:, 2]] = 1
        for i in range(4):
            src_t, dst_t = g_list[i].nonzero()
            j = (i + 1) % 4
            labels = g_list[j][src_t, dst_t]
            pair_list[i].append(np.stack([src_t, dst_t], 1))
            label_list[i].append(labels)
    pair_list = [torch.from_numpy(np.concatenate(x)).cuda() for x in pair_list]
    label_list = [torch.from_numpy(np.concatenate(x)).view(-1, 1).cuda() for x in label_list]
    return pair_list, label_list


def build_hop2_dict(train_triplets, rel_dict, dataset, path_len=2):
    cache_file = './runs/%s_path_len_%d.pickle' % (dataset, path_len)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            hop2_dict = pickle.load(f)
        return hop2_dict

    hop2_dict = {}
    src, rel, dst = train_triplets.transpose()
    rel_dict_r = {v: int(k.split('_')[0] == 'competitive') for k, v in rel_dict.items()}
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in rel_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        rel_t = rel[ids]
        dst_t = dst[ids]

        graph = {}
        for u, r, v in zip(src_t, rel_t, dst_t):
            if u not in graph:
                graph[u] = {}
            graph[u][v] = rel_dict_r[r]

        dd = {'00': {}, '01': {}, '10': {}, '11': {}}
        for u, u_N in graph.items():
            for v, r1 in u_N.items():
                for v2, r2 in graph[v].items():
                    if u != v2:
                        r12 = str(r1) + str(r2)
                        if (u, v2) not in dd[r12]:
                            dd[r12][(u, v2)] = []
                        dd[r12][(u, v2)] += [v]

        hop2_dict[key[1:]] = {}
        for k in dd.keys():
            hop2_dict[key[1:]][k] = {}
            for u, v in dd[k].items():
                if len(v) >= path_len:
                    hop2_dict[key[1:]][k][u] = v

    with open(cache_file, 'wb') as f:
        pickle.dump(hop2_dict, f)
    return hop2_dict


def generate_sampled_hetero_graphs_and_labels(triplets, num_nodes, rel_dict, hop2_dict=None, coords=None,
                                              test=False, negative_rate=10, split_size=0.8):
    sample_size = len(triplets)
    src, rel, dst = triplets.transpose()
    # negative sampling
    if not test:
        samples, labels = negative_sampling(triplets, num_nodes, negative_rate)  # TODO: for different graphs
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

    edges_dict = {}
    mids_dict = {}
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in rel_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        dst_t = dst[ids]
        edges_dict[('p', key[1:], 'p')] = (src_t, dst_t)

        if hop2_dict:
            tmp_g = np.zeros((num_nodes, num_nodes))
            for s, o in zip(src_t, dst_t):
                tmp_g[s][o] = 1

            for r, reld in hop2_dict[key[1:]].items():
                src_t, dst_t, mid_t = [], [], []
                for (u, v2), vs in reld.items():
                    for v in vs:
                        # if v in tmp_g[u] and v2 in tmp_g[v]:
                        if tmp_g[u][v] and tmp_g[v][v2]:
                            src_t += [u]
                            dst_t += [v2]
                            mid_t += [v]

                rel_key = key[1:] + r
                edges_dict[('p', rel_key, 'p')] = (src_t, dst_t)
                mids_dict[rel_key] = torch.tensor(mid_t)

    hg = dgl.heterograph(edges_dict, num_nodes_dict={'p': num_nodes})
    hg.ndata['loc'] = coords  # must be tensor type

    if not hop2_dict:
        hg = hg
        hg = dgl.to_homogeneous(hg, ndata=['loc'])
    else:
        hg.edata['mid'] = mids_dict

    if test:
        return hg
    else:
        return hg, samples, labels


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    samples = np.concatenate((pos_samples, neg_samples))
    _, ids = np.unique(samples, axis=0, return_index=True)
    ids.sort()
    samples = samples[ids]
    labels = labels[ids]
    return samples, labels


# Main evaluation function
def mrr_hit_metric(ranks, hits=[1, 3, 5, 10, 15, 20]):
    ranks += 1  # change to 1-indexed
    # mrr = np.mean(1.0 / ranks)
    results = {}
    for hit in hits:
        avg_count = np.mean((ranks <= hit))
        results['Hit@' + str(hit)] = avg_count
        if hit >= 3:
            rank_ = 1.0 / ranks
            rank_[rank_ < 1.0 / hit] = 0
            results['MRR@' + str(hit)] = np.mean(rank_)

    return results


def overall_mrr_hit_metric(data_dir, dataset, metrics_list):
    times = ['morning', 'midday', '_night', 'late-night']
    cnt_l = [0, 0, 0, 0]
    with open('%s/%s/test.txt' % (data_dir, dataset)) as f:
        for line in f.readlines():
            _, rel, _ = line.split('\t')
            for i in range(4):
                if times[i] in rel:
                    cnt_l[i] += 1
    overall_metrics = {}
    for key in list(metrics_list[0].keys()):
        scores = [metrics[key] for metrics in metrics_list]
        score = sum([scores[i] * cnt_l[i] / sum(cnt_l) for i in range(len(scores))])
        overall_metrics[key] = score
        # score_str = str(round(score, 4))
        # overall_metrics[key] = score_str+'0'*(6-len(score_str))
    return overall_metrics


class Identity_NN(nn.Module):
    def __init__(self):
        super(Identity_NN, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


def GlorotOrthogonal(tensor, scale=2.):
    torch.nn.init.orthogonal_(tensor)
    tensor.mul_(torch.sqrt(scale / ((tensor.size(0) + tensor.size(1)) * torch.var(tensor, unbiased=False))))


class DLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super(DLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.reset()

    def reset(self):
        GlorotOrthogonal(self.fc.weight.data)
        if self.bias:
            self.fc.bias.data.zero_()

    def forward_nn(self, input_feat):
        return self.activation(self.fc(input_feat))


class SpatialGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(SpatialGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = aggregate_function(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype[-2:]](
                    rel_graph,
                    src_inputs[stype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype[-2:]](
                    rel_graph,
                    inputs[stype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]


def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]


def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)


def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)


def _stack_agg_func(inputs):
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)


def _agg_func(inputs, fn):
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def aggregate_function(agg):
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
