import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle as pkl
import os
import argparse
from models import link_pred, cite_pred
from utils import utils, metric
from data import graph_loader
import numpy as np
from IPython import embed


def calc_metric(mode, pos_preds, neg_preds, k=1):
    # Edge level
    pos_pred = torch.cat(pos_preds, dim=0) # [b, 1] -> [n, 1]
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, k) # [b, 1] -> [n*k, 1] -> [n, k]
    scores = torch.cat([pos_pred.view(-1,1), neg_pred], dim = 1) # [n, k+1]
    mrr, hits1, hits3, hits10  = metric.eval_mrr(scores) # input: [b, ], [b, k]
    if mode == 'test':
        ndcg, ndcg100 = metric.eval_ndcg(scores)
    # Node level
    # val_pred, val_gt = torch.vstack(y_pred).numpy(), torch.vstack(y_gt).numpy() # [n, 5], [n, 5]
    # cor = similarity(val_pred, val_gt)
    # if mode == 'test':
    #     male, rmsle = calc_male_rmsle(val_pred, val_gt, scaler)
    
    if mode == 'test':
        return {'mrr': mrr, 'hits@1,3,10': [hits1, hits3, hits10], 'ndcg,@100': [ndcg, ndcg100]}
    else:
        return {'mrr': mrr}
    
    
def batch_run(args, mode, edge_net, edge_h, src_ids, tgt_ids, neg_bids=None):
    # src_h = edge_net.fuse_src(torch.hstack([edge_h[src_ids], node_h[src_ids].detach()]))
    # tgt_h = edge_net.fuse_tgt(torch.hstack([edge_h[tgt_ids], node_h[tgt_ids].detach()]))
    pos_out = edge_net.predictor(edge_h[src_ids], edge_h[tgt_ids]) # [b, 1]

    # gdim = gt.shape[1]
    # pred = node_net.predictor(node_h[src_ids])[:, 0:gdim]
    
    if mode == 'test':
        return pos_out.detach().cpu()
    elif mode == 'train':
        pos_loss = -torch.log(pos_out + 1e-15).mean() # [1]
        
        # neg_bids = torch.randint(0, num_nodes, src_ids.size(), dtype=torch.long, device=args.device)
        # neg_h = edge_net.fuse_tgt(torch.hstack([edge_h[neg_bids], node_h[neg_bids].detach()]))
        
        neg_out = edge_net.predictor(edge_h[src_ids], edge_h[neg_bids])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        edge_loss = pos_loss + neg_loss
        
        # node_loss = torch.mean(torch.sum(torch.square\
        #     (pred*cite_mask[0:len(pred), 0:gdim] - gt*cite_mask[0:len(pred), 0:gdim] ), dim=1))
        return edge_loss, pos_out.detach().cpu(), neg_out.detach().cpu()


@torch.no_grad()
def test(args, edge_net, valid_g, val_test_dcit):
    edge_net.eval()
    # node_net.eval()
    
    edge_h = edge_net.gnn(valid_g.x, valid_g.edge_index)
    
    valid_rst = {}
    for ii, tt in enumerate(range(args.valid_start, args.valid_end)):
        # edge preprocess
        left, right = valid_g.ts_infos[torch.where(valid_g.ts_infos[:, 0]==tt)[0][0], 1:]
        # num_nodes = valid_g.edge_index[:, left:right].max().item() + 1
        # node preprocess
        # node_h = node_net.gnn(valid_g.edge_index[:, 0:right], valid_g.x[0:num_nodes],  \
        #         torch.zeros_like(valid_g.edge_index[0, 0:right], dtype=torch.int32)) # [2, e], [n, d]
        # node_h = torch.vstack([node_h, torch.zeros((len(edge_h)-len(node_h), node_h.shape[1]), device=args.device)]) # 没有embed的节点
        # Select the edge to be tested
        left, right = val_test_dcit['ts_cuts'][ii], val_test_dcit['ts_cuts'][ii+1]
        source_edge = val_test_dcit['src'][left:right] # [e]
        target_edge = val_test_dcit['val_tgt'][left:right] # [e]
        target_neg = val_test_dcit['val_neg'][left:right, :].view(-1) # [e, k]
        
        pos_preds, neg_preds = [], []
        # y_pred, y_gt = [], []
        for idd, perm in enumerate(tqdm(DataLoader(range(source_edge.size(0)), args.bs, shuffle=False), desc=f'{tt}')):
            src_ids, tgt_ids = source_edge[perm].to(args.device), target_edge[perm].to(args.device) # [b]
            # gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][valid_g.raw_nid[src_ids]]]
            
            pos_out = batch_run(args, 'test', edge_net, edge_h, src_ids, tgt_ids)
            
            pos_preds += [pos_out]
            # y_pred.append(pred)
            # y_gt.append(gt)
        
        source = source_edge.view(-1, 1).repeat(1, args.k).view(-1)
        for perm in tqdm(DataLoader(range(source.size(0)), args.bs, shuffle=False), desc=f'{tt}'):
            src_ids, tgt_ids = source[perm].to(args.device), target_neg[perm].to(args.device) # [b]
            # gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][valid_g.raw_nid[src_ids]]]
            
            neg_out = batch_run(args, 'test', edge_net, edge_h, src_ids, tgt_ids)
            neg_preds += [neg_out] # [b, 1]

        valid_rst[tt] = calc_metric('test', pos_preds, neg_preds, args.k)
    return valid_rst
    

def train(args, edge_net, train_g, opt_edge):
    edge_net.train()
    # node_net.train()
    train_rst = {'mrr': 0, 'cor': 0, 'loss': 0}
    total_loss = total = 0
    pos_preds, neg_preds  = [], []
    
    # cite_mask = torch.zeros((args.bs, 5), dtype=torch.bool).to(args.device)
    # for ii, tt in enumerate(range(args.train_start + args.node_year, args.train_end)):
        # cite_mask[:, 0:5-(tt - (args.train_start + args.node_year))] = True
        # edge preprocess
        # left, right = train_g.ts_infos[torch.where(train_g.ts_infos[:, 0]==tt)[0][0], 1:]
    num_nodes = train_g.edge_index[:, 0:train_g.train_val_split].max().item() + 1
    
    neg_edge = utils.local_neg_sample(train_g.edge_index[:, 0:train_g.train_val_split], \
        num_nodes, 1, quick=False)[1, :, 0] # [2, e, 1] -> [e]
    
    # for perm in tqdm(DataLoader(range(source_edge.size(0)), args.bs, shuffle=False), desc=f'{tt}'):
    for perm in tqdm(DataLoader(range(0, train_g.train_val_split), args.bs, shuffle=False)):
        src_ids, tgt_ids = train_g.edge_index[0, perm], train_g.edge_index[1, perm]# [b]
        # 1. Get embeds of two graphs
        edge_h = edge_net.gnn(train_g.x[0:num_nodes], train_g.edge_index[:, 0:train_g.train_val_split])
        # node_h = node_net.gnn(train_g.edge_index[:, 0:right], train_g.x[0:num_nodes],  \
        #     torch.zeros_like(train_g.edge_index[0, 0:right], dtype=torch.int32))

        # gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][train_g.raw_nid[src_ids]]].to(args.device)
        
        edge_loss, pos_out, neg_out = batch_run(args, 'train', edge_net, \
            edge_h, src_ids, tgt_ids, neg_edge[perm].to(args.device))
        
        opt_edge.zero_grad()
        edge_loss.backward()
        opt_edge.step()
        
        # opt_node.zero_grad()
        # node_loss.backward()
        # opt_node.step()
        
        total += src_ids.size(0)
        total_loss += edge_loss.item() * src_ids.size(0)
        
        pos_preds += [pos_out]
        neg_preds += [neg_out]
        # y_pred.append(pred)
        # y_gt.append(gt.detach().cpu())
        
        # if len(pos_preds)>1:
        #     break
    
    rst = calc_metric('train', pos_preds, neg_preds, k=1)
    train_rst['mrr'] = rst['mrr']
    # train_rst['cor'] += rst['cor']
        
    train_rst['loss'] = round(total_loss / total, 4)
    # train_rst['mrr'] = round(train_rst['mrr'] / (args.train_end-(args.train_start + args.node_year)), 4)
    # train_rst['cor'] = round(train_rst['cor'] / (args.train_end-(args.train_start + args.node_year)), 4)
    return train_rst



def main(args):
    # node_infos = graph_loader.label_preprocess(args)
    all_g, val_test_dcit = graph_loader.load_graph(args)
    logger.info(f'all_g: {all_g}')
    
    edge_net = link_pred.EDGE_NET(args).to(args.device)
    # node_net = cite_pred.NODE_NET(args).to(args.device)
    opt_edge = torch.optim.Adam(edge_net.parameters(), lr=args.lr)
    # opt_node = torch.optim.Adam(node_net.parameters(), lr=args.lr)
    
    all_g.x = all_g.x.to(args.device)
    all_g.edge_index = all_g.edge_index.to(args.device)
    
    # valid_rst = test(args, edge_net, all_g, val_test_dcit)
    # logger.info(valid_rst)
    for epoch in range(args.epochs):
        train_rst = train(args, edge_net, all_g, opt_edge)
        valid_rst = test(args, edge_net, all_g, val_test_dcit)
        
        logger.info(f'epoch: {epoch}, valid: {valid_rst}, train: {train_rst}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='')
    parser.add_argument('--gid', type=int, default=-1)
    # 1. Dataset
    parser.add_argument("--dataset", "-d", type=str, default='aps')
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument('--train_start', "-ts", type=int, default=2000)
    parser.add_argument('--train_end', "-te", type=int, default=2011)
    parser.add_argument('--valid_start', "-vs", type=int, default=2011)
    parser.add_argument('--valid_end', "-ve", type=int, default=2012)
    parser.add_argument('--node_year', type=int, default=5)
    
    parser.add_argument('--train_num', type=int, default=3000)
    parser.add_argument('--num_ngh', nargs='+', default=[100, 20, 1, 15])
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--k', type=int, default=10)
    # parser.add_argument('--num_year', type=int, default=5)
    
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--bs', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0)
    
    parser.add_argument('--gnn', type=str, default="rgcn")
    parser.add_argument('--method', type=str, default="cave")
    parser.add_argument('--fusion_mode', type=str, default="cat")
    
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--only_papers', action='store_true')
    parser.add_argument('--save_dir', type=str, default="../save")
    
    parser.add_argument('--edge_gnn', type=str, default="sage")

    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if not args.data_path:
        # args.data_path = f'../../01_process/data_papers/{args.dataset}/'
        args.data_path = f'../preprocess/data/{args.dataset}/'
    
    args.num_rels = len(args.num_ngh)
     
    logger = utils.set_logger(args)
    main(args)