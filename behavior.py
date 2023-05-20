from collections import Counter
import pandas as pd
import argparse
import random
import os

random.seed(0)

days = {'Apr': '30', 'May': '31', 'Jun': '30', 'Jul': '31', 'Aug': '31', 'Sep': '30', 'Oct': '31', 'Nov': '30',
              'Dec': '31', 'Jan': '31', 'Feb': '28'}
months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
times = {}
for i in range(3, 9):
    times[str(i).zfill(2)] = '0'
for i in range(9, 15):
    times[str(i).zfill(2)] = '1'
for i in range(15, 21):
    times[str(i).zfill(2)] = '2'
for i in list(range(21, 24)) + list(range(0, 3)):
    times[str(i).zfill(2)] = '3'


def get_time(times, offset=9):
    m, d, h = times
    d, h = int(d), int(h)
    if offset > 0:
        if h + offset > 23:
            if d + 1 <= int(days[m]):
                d = d + 1
            else:
                d = 1
                m = months[months.index(m) + 1]
            h = (h + offset) % 24
        else:
            h = h + offset
    else:
        if h + offset < 0:
            if d > 1:
                d = d - 1
            else:
                m = months[months.index(m) - 1]
                d = days[m]
            h = 24 + h + offset
        else:
            h = h + offset
    d, h = str(d).zfill(2), str(h).zfill(2)
    zone = times[h]
    if h in ['00', '01', '02']:
        if d == '01':
            m = months[months.index(m) - 1]
            d = days[m]
        else:
            d = str(int(d) - 1).zfill(2)
    return int(zone), m + '#' + d


def load_observes(data):
    observerations = {}
    for index, row in data.iterrows():
        user = row['userId']
        tag = row['venueCategory']
        time = row['utcTimestamp']
        place = row['venueId']
        x, y = row['latitude'], row['longitude']
        if place not in observerations:
            observerations[place] = {'user': [], 'loc': [], 'time': []}
        observerations[place]['user'].append(user)
        observerations[place]['time'].append(time[4:13].split(' '))
        observerations[place]['loc'].append((x, y, tag))


def update_observ_time(observerations):
    obs_new = {}
    for k, v in observerations.items():
        if len(set(v['user'])) >= 4:
            obs_new[k] = [set() for _ in range(4)]
            for time, user in zip(v['time'], v['user']):
                zone, md = get_time(time)
                obs_new[k][zone].add(md + '#' + str(user))
    return obs_new


def build_interaction(obs_new):
    # basic correls
    rel = [[] for i in range(4)]
    for k1, v1 in obs_new.items():
        for k2, v2 in obs_new.items():
            if k1 == k2:
                continue
            for i in range(4):
                v12 = v1[i] & v2[i]
                if len(v12) > 2:
                    rel[i] += [(k1, k2, len(v12))]

    # processing location nodes
    node_set = set()
    for rr in rel:
        for u, v, n in rr:
            if n < 4: continue
            node_set.add(u)
            node_set.add(v)
    poi_dict = {}
    for k in node_set:
        poi_dict[k] = {}
        xy_tag = observerations[k]['loc']
        xy = list(set([(x, y) for x, y, t in xy_tag]))
        tag = list(set([t for x, y, t in xy_tag]))
        xy_count = Counter([(x, y) for x, y, t in xy_tag])
        tag_count = Counter([t for x, y, t in xy_tag])
        xy_count = sorted(xy_count.items(), key=lambda x: x[1], reverse=True)
        tag_count = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        if len(xy_count) == 1:
            poi_dict[k]['coord'] = xy_count[0][0]
        else:
            poi_dict[k]['coord'] = xy_count[0][0]
            pass
        if len(tag_count) == 1:
            poi_dict[k]['tag'] = tag_count[0][0]
        else:
            poi_dict[k]['tag'] = tag_count[0][0]

    threshold = 4
    all_pair_set_dict = {'time' + str(i + 1): set() for i in range(4)}
    for i in range(4):
        for u, v, n in rel[i]:
            if n < threshold:
                continue
            uv = v + '#' + u if u > v else u + '#' + v
            all_pair_set_dict['time' + str(i + 1)].add(uv)
    for i in range(4):
        for j in range(i + 1, 4):
            s1 = all_pair_set_dict['time' + str(i + 1)]
            s2 = all_pair_set_dict['time' + str(j + 1)]
            print(len(s1), len(s2), str(i + 1) + '#' + str(j + 1), 1.0 * len(s1 & s2) / len(s1 | s2),
                  1.0 * len(s1 & s2) / min(len(s1), len(s2)))
    interactions_set_all = set()
    interactions_all = {}
    for k, v in all_pair_set_dict.items():
        interactions_set_all = interactions_set_all | v
        interactions_all[k] = list(v)

    return poi_dict, rel, interactions_all


def build_dataset(nd, rel, all_pair_list_d):
    all_pair_list_d['time0'] = [x for x in all_pair_list_d['time4']]
    ratio = 0.2
    d_val = {'time' + str(i + 1): set() for i in range(4)}
    threshold = 4
    rel = dict()
    for i in range(4):
        for u, v, n in rel[i]:
            if n < threshold:
                continue
            if v not in rel:
                rel[v] = {'time1': {}, 'time2': {}, 'time3': {}, 'time4': {}}
            rel[v]['time' + str(i + 1)][u] = n

    rel_g_skip = dict()
    for k1, v in rel.items():
        rel_g_skip[k1] = {}
        for t, v2 in v.items():
            if len(v2) == 0:
                rel_g_skip[k1][t] = ''
                continue
            k2_list = [k2 for k2 in v2]
            ind = random.randint(0, len(k2_list) - 1)
            k2 = k2_list[ind]
            rel_g_skip[k1][t] = k2

    for i in range(4):
        i = str(i + 1)
        lll = all_pair_list_d['time' + i]
        num_test = int(len(lll) * ratio)
        while len(d_val['time' + i]) < num_test:
            ind = random.randint(0, len(lll) - 1)
            if lll[ind] in d_val['time' + i]:
                continue
            b1, b2 = lll[ind].split('#')
            block = False
            for j in range(int(i), 5):
                if len(rel[b1]['time' + str(j)]) == 1 or len(rel[b2]['time' + str(j)]) == 1:
                    block = True
                    break
            if block: continue
            for j in range(int(i), 5):
                if b2 not in rel[b1]['time' + str(j)]:
                    continue
                d_val['time' + str(j)].add(lll[ind])
                rel[b1]['time' + str(j)].pop(b2)
                rel[b2]['time' + str(j)].pop(b1)

    tp, tpd, vp = {'time0': []}, {'time0': []}, {'time0': []}
    for i in range(1, 5):
        time = 'time' + str(i)
        ll = list(d_val[time])
        random.shuffle(ll)

        times = ['morning', 'midday', 'night', 'late-night']
        tpd[time] = [x + '#' + times[i - 1] for x in ll[:int(len(ll) / 2)]]
        vp[time] = [x + '#' + times[i - 1] for x in ll[int(len(ll) / 2):]]
        tp[time] = [x + '#' + times[i - 1] for x in list(set(all_pair_list_d[time]) - set(ll))]
        tpd['time0'] += tpd[time]
        tp['time0'] += tp[time]
        vp['time0'] += vp[time]

    nd = dict()
    for rr in rel:
        for u, v, n in rr:
            if n < threshold: continue
            if u not in nd:
                nd[u] = nd[u]
            if v not in nd:
                nd[v] = nd[v]

    return nd, tp, vp, tpd


def save_interaction_dataset(output_path, node_dict, train_pair_dict, valid_pair_dict, test_pair_dict):
    times = ['morning', 'midday', 'night', 'late-night']
    with open('%s/entities.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(str(i) + '\t' + bid + '\n')
    with open('%s/coords.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(bid + '\t' + str(value['coord'][0]) + '\t' + str(value['coord'][1]) + '\n')

    rel_types = ['competitive', 'complementary']
    idx = 0
    with open('%s/rels.dict' % output_path, 'w') as f:
        for r in rel_types:
            for t in times:
                f.write(str(idx) + '\t' + '_'.join([r, t]) + '\n')
                idx += 1
    f1 = open('%s/train.txt' % output_path, 'w')
    f2 = open('%s/valid.txt' % output_path, 'w')
    f3 = open('%s/test.txt' % output_path, 'w')
    cnt1, cnt2, cnt3 = 0, 0, 0
    for uv in train_pair_dict['time0']:
        u, v, t = uv.split('#')
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt1 += diff_tag
        rel = rel_types[diff_tag]
        f1.write(u + '\t' + '_'.join([rel, t]) + '\t' + v + '\n')
        f1.write(v + '\t' + '_'.join([rel, t]) + '\t' + u + '\n')
    for uv in valid_pair_dict['time0']:
        u, v, t = uv.split('#')
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt2 += diff_tag
        rel = rel_types[diff_tag]
        f2.write(u + '\t' + '_'.join([rel, t]) + '\t' + v + '\n')
        f2.write(v + '\t' + '_'.join([rel, t]) + '\t' + u + '\n')
    for uv in test_pair_dict['time0']:
        u, v, t = uv.split('#')
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt3 += diff_tag
        rel = rel_types[diff_tag]
        f3.write(u + '\t' + '_'.join([rel, t]) + '\t' + v + '\n')
        f3.write(v + '\t' + '_'.join([rel, t]) + '\t' + u + '\n')
    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':
    reader = argparse.ArgumentParser()
    reader.add_argument('--raw-in', type=str, default='./data/hamburg_2023-03-01-gti.csv')
    reader.add_argument('--out', type=str, default='./data/hamburg/')
    args = reader.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    data = pd.read_csv(args.input_data)
    observerations=load_observes(data)
    observerations_time = update_observ_time(observerations)
    node_dict, rel, all_pair_list_d = build_interaction(observerations_time)
    node_dict, train, valid, test = build_dataset(node_dict, rel, all_pair_list_d)

    save_interaction_dataset(args.output_path, node_dict, train, valid, test)