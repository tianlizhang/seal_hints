import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from IPython import embed


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels))
        for l in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels))
        self.convs.append(
            GATConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    

class EDGE_NET(torch.nn.Module):
    def __init__(self, args):
        super(EDGE_NET, self).__init__()
        print(f'edge_gnn = {args.edge_gnn}')
        if args.edge_gnn == 'gcn':
            self.gnn = GCN(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        if args.edge_gnn == 'gat':
            self.gnn = GAT(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        else:
            self.gnn = SAGE(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        
        self.predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, \
            args.num_layers, args.dropout).to(args.device)
        
        self.fusion_mode = args.fusion_mode
        print('fusin_mode', self.fusion_mode)
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
    
    
    def cat_fusion(self, x_src, imputed_embs, src_ids):
        hh = x_src.shape[1]
        x_out = torch.hstack([x_src, x_src]) # [b, 2*h]
        imputed_out = torch.hstack([imputed_embs, imputed_embs]) # [s, 2*h]
        
        cnt, lst = 0, []
        # embed()
        for i in range(len(x_out)):
            if i and src_ids[i-1] != src_ids[i]:
                imputed_out[cnt, hh:] = torch.mean(torch.stack(lst), dim=0)
                lst = []
                cnt += 1
            x_out[i, hh:] = imputed_embs[cnt]
            lst.append(x_src[i])
            
        x_out = self.cat_linear1(x_out)
        imputed_out = self.cat_linear2(imputed_out)
        return x_out, imputed_out

