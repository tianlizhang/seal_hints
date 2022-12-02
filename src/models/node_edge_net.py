import torch
import torch.nn as nn
from .gnn import GCN, GAT, SAGE, RGCN
from .link_pred import LinkPredictor
from .node_net import FcPredictor, CitePredictor


class NEN(nn.Module): # NODE EDGE NET
    def __init__(self, args):
        super(NEN, self).__init__()
        self.device = args.device
        self.alpha = args.alpha
        print(f'edge_gnn = {args.edge_gnn}')
        if args.edge_gnn == 'gcn':
            self.edge_gnn = GCN(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        if args.edge_gnn == 'gat':
            self.edge_gnn = GAT(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        else:
            self.edge_gnn = SAGE(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
            
        self.edge_predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, \
            args.num_layers, args.dropout).to(args.device)
        
        print(f'node_gnn = {args.node_gnn}')
        self.ngnn = args.node_gnn
        if args.node_gnn == 'gcn':
            self.node_gnn = GCN(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        elif args.node_gnn == 'gat':
            self.node_gnn = GAT(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        elif args.node_gnn == 'rgcn':
            self.node_gnn = RGCN(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, 1, args.dropout).to(args.device) 
        else:
            self.node_gnn = SAGE(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)

        print(f'node_predictor = {args.node_predictor}')
        if args.node_predictor == 'fc':
            self.node_predictor = FcPredictor(args.hidden_channels, args.hidden_dims, \
                args.node_year, args.device).to(args.device)
            
        else:
            self.node_predictor = CitePredictor(args.hidden_channels, args.hidden_dims, \
                args.node_year, args.device).to(args.device)
        
        print(f'fusin_mode = {args.fusion_mode}')
        self.fusion_mode = args.fusion_mode
        # if self.fusion_mode == 'attn':
        #     self.num_mha = args.num_mha
        #     self.mha_lst1 = nn.ModuleList([MultiHeadAttention(args.hidden_channels, args.num_head)
        #                         for _ in range(self.num_mha)])
        #     self.mha_lst2 = nn.ModuleList([MultiHeadAttention(args.hidden_channels, args.num_head)
        #                         for _ in range(self.num_mha)])
        if self.fusion_mode == 'gru':
            self.gru = nn.GRU(args.hidden_channels, args.hidden_channels, batch_first=True)
        elif 'cat' in self.fusion_mode:
            self.fuse_src = nn.Linear(2*args.hidden_channels, args.hidden_channels)
            self.fuse_tgt = nn.Linear(2*args.hidden_channels, args.hidden_channels)
    
    
    def forward(self, train_g, src_ids, tgt_ids, neg_ids, right, num_nodes, gt, cite_mask):
        edge_h = self.edge_gnn(train_g.x, train_g.edge_index)
        if self.ngnn == 'rgcn':
            node_h = self.node_gnn(train_g.x[:, 0:num_nodes], train_g.edge_index[0:right],  \
                torch.zeros_like(train_g.edge_index[0, 0:right], dtype=torch.int32))
        else:
            node_h = self.node_gnn(train_g.x[0:num_nodes], train_g.edge_index[:, 0:right])
        
        edge_loss, node_loss, pos_out, neg_out, pred = self.batch_run('train', edge_h, node_h, \
            src_ids, tgt_ids, gt, cite_mask, neg_ids)

        loss = self.alpha * edge_loss + node_loss
        return loss, pos_out, neg_out, pred
        
    
    def batch_run(self, mode, edge_h, node_h, src_ids, tgt_ids, gt, cite_mask=None, neg_ids=None):
        src_h = self.fuse_src(torch.hstack([edge_h[src_ids], node_h[src_ids].detach()]))
        tgt_h = self.fuse_tgt(torch.hstack([edge_h[tgt_ids], node_h[tgt_ids].detach()]))
        pos_out = self.edge_predictor(src_h, tgt_h) # [b, 1]
        
        gdim = gt.shape[1]
        pred = self.node_predictor(node_h[src_ids])[:, 0:gdim]
        
        if mode == 'test':
            return pos_out.detach().cpu(), pred.detach().cpu()
        elif mode == 'train':
            pos_loss = -torch.log(pos_out + 1e-15).mean() # [1]
            
            neg_h = self.fuse_tgt(torch.hstack([edge_h[neg_ids], node_h[neg_ids].detach()]))
            
            neg_out = self.edge_predictor(src_h, neg_h)
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            edge_loss = pos_loss + neg_loss
            
            node_loss = torch.mean(torch.sum(torch.square\
                (pred*cite_mask[0:len(pred), 0:gdim] - gt*cite_mask[0:len(pred), 0:gdim] ), dim=1))
            return edge_loss, node_loss, pos_out.detach().cpu(), neg_out.detach().cpu(), pred.detach().cpu()