import numpy as np
import os
from dgl import backend as F


class RelDS(object):
    def __init__(self, name, raw_dir, grid_len, verbose=True):
        self.name = name
        self.grid_len = grid_len
        self.verbose = verbose
        self.raw_path = os.path.join(raw_dir, self.name)
        self.process()

    def process(self):
        root_path = self.raw_path
        entity_path = os.path.join(root_path, 'entities_in.dict')
        rel_path = os.path.join(root_path, 'rels_in.dict')
        grid_path = os.path.join(root_path, 'grids_%d.dict' % self.grid_len)
        pc = os.path.join(root_path, 'c.dict')
        pt = os.path.join(root_path, 'train_in.txt')
        pv = os.path.join(root_path, 'valid_in.txt')
        pt1 = os.path.join(root_path, 'test_in.txt')
        ed = _read_d(entity_path)
        rel = _read_d(rel_path)
        c = _read_c(pc, ed)
        grid = _read_gg(grid_path, ed)
        training = np.asarray(_read_trip(pt, ed, rel))
        validaiton = np.asarray(_read_trip(pv, ed, rel))
        test = np.asarray(_read_trip(pt1, ed, rel))
        num_nodes = len(ed)
        num_rels = len(rel)
        self._rel_dict = rel
        self._train = training
        self._valid = validaiton
        self._test = test
        self._grid = grid

        self._coords = c
        self._num_nodes = num_nodes
        self._num_rels = num_rels
        self._entity_dict = ed

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return self._num_nodes

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def rel_dict(self):
        return self._rel_dict

    @property
    def coords(self):
        return self._coords

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

    @property
    def grid(self):
        return self._grid


def _read_gg(filename, entity_dic):
    gg = {}
    with open(filename) as f:
        for line in f.readlines():
            gai = line.strip().split('\t')
            k = tuple(map(int, gai[0].split(',')))
            gg[k] = [entity_dic[bid] for bid in gai[1:]]
    return gg


def _read_c(filename, entity_dict):
    c = [[] for _ in range(len(entity_dict))]
    with open(filename) as f:
        for l in f.readlines():
            l = l.strip().split('\t')
            c[entity_dict[l[0]]] = [float(l[1]), float(l[2])]
    return F.tensor(c)


def _read_d(filename):
    d = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def _read(filename):
    with open(filename) as f:
        for line in f.readlines():
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_trip(filename, entity_dict, rel_dict):
    l = []
    for triplet in _read(filename):
        s = entity_dict[triplet[0]]
        r = rel_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l
