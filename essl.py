import argparse
import numpy as np
import torch
import random
from ds import RelDS
from layers import EGNN
import layers as utils

torch.set_num_threads(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    data = RelDS(name=args.dataset, grid_len=args.grid_len, raw_dir=args.data_dir)
    num_nodes = data.num_nodes
    da = data.train
    c = data.coords
    reld = data.rel_dict
    nrel = data.num_rels
    gd = data.grid

    scopes = utils.dynamic_dist(args.bin_num, c, da)

    log = str(args) + '\n'

    model = EGNN(num_nodes, args.n_hidden, nrel, scope_i=scopes, neig=args.n_neighbor, dout=args.dropout,
                 rp1=args.global_weight, d1_i=args.local_weight)

    evo_data, evo_labels = utils.build_dynamic_labels(da, reld, num_nodes)
    hop2_dict = utils.build_hop2_dict(da, reld, args.dataset, path_len=2)
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(da, num_nodes, reld, hop2_dict,
                                                                 coords=c, test=True)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    if use_cuda:
        model.cuda()
        test_graph = test_graph.to(args.gpu)
        node_id = node_id.cuda()
        train_data_list = utils.select_time_data(torch.from_numpy(da).cuda(), reld)

    g = test_graph
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)
    model_state_file = './output/%s.pth' % (args.run_name)

    # training loop
    print("start training...")
    epoch = 0
    min_loss = 1e9
    stop_epoch = 0
    while True:
        model.train()
        epoch += 1
        grs, pos, negs, grl = utils.generate_batch_grids(gd, args.global_batch_size, args.global_neg_ratio,
                                                         sampling_hop=2)
        grs = torch.from_numpy(grs).long()
        pos, negs = torch.from_numpy(pos).long(), torch.from_numpy(negs).long()
        grl = torch.from_numpy(grl)

        if use_cuda:
            grs = grs.cuda()
            pos = pos.cuda()
            negs = negs.cuda()
            grl = grl.cuda()
            coord = c[node_id].cuda()

        f_in = model(g, node_id)
        loss = model.get_loss(g, f_in, evo_data, evo_labels, grs, pos, negs, grl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f}".format(epoch, train_loss))
        log += "Epoch {:04d} | Loss {:.4f} \n".format(epoch, train_loss)
        optimizer.zero_grad()

        if train_loss < min_loss:
            min_loss = train_loss
            stop_epoch = 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        else:
            stop_epoch += 1
        if stop_epoch == 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trial')
    parser.add_argument("--dout", type=float, default=0.3)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-neighbor", type=int, default=7)
    parser.add_argument("--n-epochs", type=int, default=3000)
    parser.add_argument("-d", "--ds", type=str, default='hh')
    parser.add_argument("--trained-model", type=str, default='')
    parser.add_argument("--size-grid", type=int, default=450)
    parser.add_argument("--global-neg-loss-ratio", type=int, default=3)
    parser.add_argument("--global", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--locality", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--negative-sample", type=int, default=5)
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--distance-buckets", type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    run_name = f'essl'
    run_name = args.name + "_" + run_name
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)
