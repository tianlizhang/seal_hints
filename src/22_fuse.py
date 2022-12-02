import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from models import node_edge_net
from utils import utils, metric
from data import graph_loader
import numpy as np
from IPython import embed


@torch.no_grad()
def test(args, model, valid_g, val_test_dcit, node_infos):
    model.eval()
    edge_h = model.edge_gnn(valid_g.x, valid_g.edge_index)
    
    valid_rst = {}
    for ii, tt in enumerate(range(args.valid_start, args.valid_end)):
        # edge preprocess
        left, right = valid_g.ts_infos[torch.where(valid_g.ts_infos[:, 0]==tt)[0][0], 1:]
        num_nodes = valid_g.edge_index[:, left:right].max().item() + 1
        # node preprocess
        if args.node_gnn == 'rgcn':
            node_h = model.node_gnn(valid_g.x[0:num_nodes], valid_g.edge_index[:, 0:right], \
                torch.zeros_like(valid_g.edge_index[0, 0:right], dtype=torch.int32)) # [2, e], [n, d]
        else:
            node_h = model.node_gnn(valid_g.x[0:num_nodes], valid_g.edge_index[:, 0:right]) # [2, e], [n, d]
        node_h = torch.vstack([node_h, torch.zeros((len(edge_h)-len(node_h), node_h.shape[1]), device=args.device)]) # 没有embed的节点
        # Select the edge to be tested
        left, right = val_test_dcit['ts_cuts'][ii], val_test_dcit['ts_cuts'][ii+1]
        source_edge = val_test_dcit['src'][left:right] # [e]
        target_edge = val_test_dcit['val_tgt'][left:right] # [e]
        target_neg = val_test_dcit['val_neg'][left:right, :].view(-1) # [e, k]
        
        pos_preds, neg_preds = [], []
        y_pred, y_gt = [], []
        for idd, perm in enumerate(tqdm(DataLoader(range(source_edge.size(0)), args.bs, shuffle=False), desc=f'{tt}')):
            src_ids, tgt_ids = source_edge[perm].to(args.device), target_edge[perm].to(args.device) # [b]
            gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][valid_g.raw_nid[src_ids]]]
            
            pos_out, pred = model.batch_run('test', edge_h, node_h, src_ids, tgt_ids, gt)
            # pos_out, pred = batch_run(args, 'test', edge_net, node_net, edge_h, node_h, src_ids, tgt_ids, gt)
            pos_preds += [pos_out]
            y_pred.append(pred)
            y_gt.append(gt)
        
        source = source_edge.view(-1, 1).repeat(1, args.k).view(-1)
        for perm in tqdm(DataLoader(range(source.size(0)), args.bs, shuffle=False), desc=f'{tt}'):
            src_ids, tgt_ids = source[perm].to(args.device), target_neg[perm].to(args.device) # [b]
            gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][valid_g.raw_nid[src_ids]]]
            
            neg_out, pred = model.batch_run('test', edge_h, node_h, src_ids, tgt_ids, gt)
            # neg_out, pred = batch_run(args, 'test', edge_net, node_net, edge_h, node_h, src_ids, tgt_ids, gt)
            neg_preds += [neg_out] # [b, 1]

        valid_rst[tt] = metric.calc_metric('test', pos_preds, neg_preds, y_pred, y_gt, node_infos[tt]['scaler'], args.k)
    return valid_rst


def train(args, model, train_g, node_infos, opt):
    model.train()
    train_rst = {'mrr': 0, 'cor': 0, 'loss': 0}
    total_loss = total = 0
    pos_preds, neg_preds, y_pred, y_gt  = [], [], [], []
    
    cite_mask = torch.zeros((args.bs, 5), dtype=torch.bool).to(args.device)
    for ii, tt in enumerate(range(args.train_start + args.node_year, args.train_end)):
        cite_mask[:, 0:5-(tt - (args.train_start + args.node_year))] = True
        # edge preprocess
        left, right = train_g.ts_infos[torch.where(train_g.ts_infos[:, 0]==tt)[0][0], 1:]
        num_nodes = train_g.edge_index[:, left:right].max().item() + 1
        
        neg_edge = utils.local_neg_sample(train_g.edge_index[:, left:right], \
            num_nodes, 1, quick=False)[1, :, 0] # [2, e, 1] -> [e]
        
        # for perm in tqdm(DataLoader(range(source_edge.size(0)), args.bs, shuffle=False), desc=f'{tt}'):
        for perm in tqdm(DataLoader(range(left, right), args.bs, shuffle=False), desc=f'{tt}'):
            src_ids, tgt_ids = train_g.edge_index[0, perm], train_g.edge_index[1, perm]# [b]
            gt = node_infos[tt]['labels_norm'][node_infos[tt]['sid2pos'][train_g.raw_nid[src_ids]]].to(args.device)
            neg_ids = neg_edge[perm-left].to(args.device)
            
            loss, pos_out, neg_out, pred= model(train_g, src_ids, tgt_ids, neg_ids, right, num_nodes, gt, cite_mask)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total += src_ids.size(0)
            total_loss += loss.item() * src_ids.size(0)
            
            pos_preds += [pos_out]
            neg_preds += [neg_out]
            y_pred.append(pred)
            y_gt.append(gt.detach().cpu())
        
        rst = metric.calc_metric('train', pos_preds, neg_preds, y_pred, y_gt, node_infos[tt]['scaler'], k=1)
        train_rst['mrr'] += rst['mrr']
        train_rst['cor'] += rst['cor']
    
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
    train_rst['loss'] = round(total_loss / total, 4)
    train_rst['mrr'] = round(train_rst['mrr'] / (args.train_end-(args.train_start + args.node_year)), 4)
    train_rst['cor'] = round(train_rst['cor'] / (args.train_end-(args.train_start + args.node_year)), 4)
    return train_rst


def main(args):
    node_infos = graph_loader.label_preprocess(args)
    all_g, val_test_dcit = graph_loader.load_graph(args)
    logger.info(f'all_g: {all_g}')
    
    model = node_edge_net.NEN(args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    all_g.x = all_g.x.to(args.device)
    all_g.edge_index = all_g.edge_index.to(args.device)
    
    # valid_rst = test(args, edge_net, node_net, all_g, val_test_dcit, node_infos)
    # logger.info(valid_rst)
    for epoch in range(args.epochs):
        train_rst = train(args, model, all_g, node_infos, opt)
        valid_rst = test(args, model, all_g, val_test_dcit, node_infos)
        
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
    parser.add_argument('--hidden_dims', nargs='+', default=[20, 10, 20, 8])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0)
    
    parser.add_argument('--edge_gnn', type=str, default="sage")
    parser.add_argument('--node_gnn', type=str, default="sage")
    parser.add_argument('--node_predictor', type=str, default="cites")
    parser.add_argument('--fusion_mode', type=str, default="cat")
    
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--only_papers', action='store_true')
    parser.add_argument('--save_dir', type=str, default="../save")
    
    

    args = parser.parse_args()
    utils.set_random_seed(seed=42)
    args.device = utils.get_free_device(args)
    
    if not args.data_path:
        # args.data_path = f'../../01_process/data_papers/{args.dataset}/'
        args.data_path = f'../preprocess/data/{args.dataset}/'
    
    args.num_rels = len(args.num_ngh)
     
    logger = utils.set_logger(args)
    main(args)