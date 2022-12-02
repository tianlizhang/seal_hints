import os, dgl, torch
import pickle as pkl
from tqdm import tqdm, trange
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from utils import utils
from IPython import embed


def label_preprocess(args):
    labels_cum_log = pkl.load(open(os.path.join(args.data_path, 'labels_cum_log.pkl'), 'rb'))
    node_infos = {}
    for year in range(args.train_start, args.valid_end):
        sids_in_ngh = labels_cum_log[year]['nid'].tolist()
        key_nodes = torch.tensor(sids_in_ngh).unique()  # 按照label的顺序排列成
        sid2pos = -torch.ones(max(key_nodes) + 1, dtype=torch.long) # [108073]
        sid2pos[key_nodes] = torch.arange(len(key_nodes))
        
        lb = labels_cum_log[year].iloc[:, 2:2+args.node_year].values
        lb_norm, scaler = utils.label_normorlization(lb)
        labels_norm = torch.from_numpy(lb_norm)
        
        node_infos[year] = {'labels_norm': labels_norm, 'scaler': scaler, 'sid2pos': sid2pos}
    return node_infos


def load_graph(args):
    graph = dgl.load_graphs(f'../preprocess/data/{args.dataset}/graph.bin')[0][0]
    ts_eids = graph.filter_edges(lambda x: (x.data['ts']>=args.train_start) & (x.data['ts']<args.valid_end))
    ts_graph = dgl.edge_subgraph(graph, ts_eids)
    
    gpath = f'../preprocess/data/{args.dataset}/graph_dict_{args.k}k_{args.train_start}_{args.train_end}_{args.valid_end}.pkl'
    if os.path.exists(gpath):
        print(f'Loading graph infos from {gpath}')
        out = pkl.load(open(gpath, 'rb'))
        
        all_g = build_all_graph(args, ts_graph, out['rest_eind'], out['rest_tsp'])
        return all_g, out['val_test_dict']
    else:
        val_test_dict, rest_eind, rest_tsp = two_cites_as_valid_test(args, ts_graph)
        all_g = build_all_graph(args, ts_graph, rest_eind, rest_tsp)
        
        val_test_dict['val_neg'] = neg_sample(args, val_test_dict['src'], val_test_dict['val_tgt'], all_g.num_nodes)
        val_test_dict['test_neg'] = neg_sample(args, val_test_dict['src'], val_test_dict['test_tgt'], all_g.num_nodes)

        out = {'rest_eind': rest_eind, 'rest_tsp': rest_tsp, 'val_test_dict': val_test_dict}
        pkl.dump(out, open(gpath, 'wb'))
        return all_g, val_test_dict
 

def two_cites_as_valid_test(args, ts_graph):
    valid_eids = ts_graph.filter_edges(lambda x: x.data['ts'] >= args.valid_start)
    valid_src = ts_graph.edges()[0][valid_eids].numpy()
    valid_tgt = ts_graph.edges()[1][valid_eids].numpy()
    valid_tsp = ts_graph.edata['ts'][valid_eids].numpy()
    
    pos_cuts = [0] + [pos for pos in range(len(valid_src)) if pos and valid_src[pos-1] != valid_src[pos]] + [len(valid_src)]

    val_test_src, val_tgt, test_tgt,  = [], [], []
    val_ts, rest_pos = [], []
    for ii in trange(len(pos_cuts)-1, desc='two_cites_as_valid_test'):
        left, right = pos_cuts[ii], pos_cuts[ii+1]
        if right - left < 3:
            continue
        
        val_test_src.append(valid_src[left])
        val_tgt.append(valid_tgt[left])
        test_tgt.append(valid_tgt[left+1])
        val_ts.append(valid_tsp[left])
        
        rest_pos.extend([pos for pos in range(left+2, right)])
    rest_eind = torch.stack([torch.tensor(valid_src[rest_pos]), torch.tensor(valid_tgt[rest_pos])]) # [2, e1]
    rest_tsp = valid_tsp[rest_pos]
    
    _, ts_cuts = np.unique(np.array(val_ts), return_index=True) # include 0 
    ts_cuts = list(ts_cuts) + [len(val_ts)]
    
    val_test_dict = {'src': torch.tensor(val_test_src), 'val_tgt':  torch.tensor(val_tgt), \
        'test_tgt': torch.tensor(test_tgt), 'ts_cuts': ts_cuts}
    return val_test_dict, rest_eind, rest_tsp
    
    
def build_all_graph(args, ts_graph, rest_eind, rest_tsp):
    train_eids = ts_graph.filter_edges(lambda x: x.data['ts'] < args.train_end)
    train_eind = torch.stack(ts_graph.edges())[:, train_eids] # [2, e2]
    train_tsp = ts_graph.edata['ts'][train_eids]
    
    valid_eind = torch.hstack([train_eind, rest_eind]) # [2, e1+e2]
    valid_tsp = torch.hstack([train_tsp, torch.tensor(rest_tsp)])
    
    # train_graph = build_geo_graph(ts_graph, train_eind, train_tsp)
    all_graph = build_geo_graph(ts_graph, valid_eind, valid_tsp)
    all_graph.train_val_split = train_eind.shape[1]
    return all_graph
    
    
def build_geo_graph(ts_graph, eind, tsp):
    num_nodes = ts_graph.number_of_nodes()
    x = ts_graph.ndata['feat'][0:num_nodes, :]
    raw_nid = ts_graph.ndata['raw_nid'][0:num_nodes]
    
    ts_vals, ts_cuts = np.unique(tsp.numpy(), return_index=True)
    ts_cuts = list(ts_cuts) + [len(tsp.numpy())]

    num_ts = len(ts_vals)
    ts_infos = np.stack([ts_vals, ts_cuts[0:num_ts], ts_cuts[1:num_ts+1]]).transpose() #[num_ts,3] [ts, ts_start, ts_end]
    return Data(num_nodes=int(num_nodes), edge_index = eind, x=x, edge_attr=tsp, \
        ts_infos=torch.tensor(ts_infos), raw_nid=raw_nid)


def neg_sample(args, src, tgt, num_nodes):
    print('Negative sampling ...')
    pos_edges = torch.stack([src, tgt]) # [2, e]
    neg_edges = utils.local_neg_sample(pos_edges, num_nodes, args.k, quick=False) # [2, e, k]
    return neg_edges[1, :, :] # [e, k]