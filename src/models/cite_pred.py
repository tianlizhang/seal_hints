import pickle as pkl
import os
import torch
import numpy as np
import random 

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, SAGEConv
from torch.distributions.normal import Normal


class NODE_NET(nn.Module):
    def __init__(self, args):
        super(NODE_NET, self).__init__()
        # self.convs = torch.nn.ModuleList()
        # self.convs.append()
        self.gnn = RGCN_embedding(args).to(args.device)
        
        if args.method == 'fc':
            self.predictor = FC(args).to(args.device)
            print('method', args.method)
        else:
            self.predictor = CVAE(args).to(args.device)
        self.beta = args.beta
        
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
            # self.cat_linear2 = nn.Linear(2*args.hidden_channels, args.hidden_channels)
            

    def forward(self, einds, feats, rels, aligs, nghs):
        embeddings = self.rgcn(einds, feats, rels)
        align_loss = self.rgcn.align_loss(embeddings, aligs)
        
        imputed_embs = self.impute(embeddings, nghs)
        pred = self.cave(imputed_embs) # [b, 5]
        return pred, align_loss
    
    


class RGCN_embedding(nn.Module):
    def __init__(self, args):
        super(RGCN_embedding, self).__init__()
        self.num_year = args.node_year
        self.out_dims = {
            "out_dim1":64,
            "out_dim2":128,
        }
        self.in_dim = args.embed_dim
        self.num_relations = args.num_rels
        if args.gnn == 'gcn':
            self.gnn = GCN(self.in_dim , self.out_dims, self.num_relations)
        elif args.gnn == 'sage':
            self.gnn = SAGE(self.in_dim , self.out_dims, self.num_relations)
        else:
            self.gnn = RGCN(self.in_dim , self.out_dims, self.num_relations)

    def forward(self, einds, feats, rels):
        return self.gnn(feats, einds, rels)
    
    def forward_old(self, einds, feats, rels):
        embeddings = []
        for i in range(self.num_year):
            embeddings.append(self.gnn(feats[i], einds[i], rels[i]))
             # list(array([n, 128])*5)
        return embeddings
    
    def align_loss(self, embeddings, aligs):
        align_loss = 0
        for i in range(self.num_year-1):
            align_embeds1 = torch.index_select(embeddings[i], 0, index = aligs[i][0, :]) # [a, d]
            align_embeds2 = torch.index_select(embeddings[i+1], 0, index =aligs[i][1, :]) # [a, d]
            align_loss +=  (1/(self.num_year-1)) * torch.norm(align_embeds1 - align_embeds2, p=2) \
                / len(align_embeds1)
        return align_loss
             
                
class RGCN(nn.Module):
    def __init__(self, in_dim, out_dims, num_relations, dropout=0.2):
        super(RGCN, self).__init__()
        self.gcn1 = RGCNConv(in_dim, out_dims['out_dim1'], num_relations=num_relations)
        self.gcn2 = RGCNConv(out_dims['out_dim1'], out_dims['out_dim2'], num_relations=num_relations)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_type):
        H_1 = self.gcn1(x, edge_index, edge_type)
        H_1 = F.relu(H_1)
        H_1 = F.dropout(H_1, p=self.dropout)
        H_2 = self.gcn2(H_1, edge_index, edge_type)
        return H_2    


class GCN(nn.Module):
    def __init__(self, in_dim, out_dims, num_relations=None, dropout=0.2):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(in_dim, out_dims['out_dim1'])
        self.gcn2 = GCNConv(out_dims['out_dim1'], out_dims['out_dim2'])
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_type=None):
        H_1 = self.gcn1(x, edge_index)
        H_1 = F.relu(H_1)
        H_1 = F.dropout(H_1, p=self.dropout)
        H_2 = self.gcn2(H_1, edge_index)
        return H_2
    

class SAGE(nn.Module):
    def __init__(self, in_dim, out_dims, num_relations=None, dropout=0.2):
        super(SAGE, self).__init__()
        self.gcn1 = SAGEConv(in_dim, out_dims['out_dim1'])
        self.gcn2 = SAGEConv(out_dims['out_dim1'], out_dims['out_dim2'])
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_type=None):
        H_1 = self.gcn1(x, edge_index)
        H_1 = F.relu(H_1)
        H_1 = F.dropout(H_1, p=self.dropout)
        H_2 = self.gcn2(H_1, edge_index)
        return H_2
    
        

class CVAE(nn.Module):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.pred_year = args.node_year # 5
        self.device = args.device
        self.hidden = {"encoder_1":50,"encoder_2":10,
                           "decoder_1":20,"decoder_2":8,"decoder_3":1,
                           "rnn":50,"conditional_1":20}
        
        self.gru = nn.GRU(args.embed_dim, self.hidden['rnn'], batch_first=True)
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(args.hidden_channels, self.hidden['conditional_1']), 
            nn.ReLU(), 
            nn.Linear(self.hidden['conditional_1'], self.hidden['encoder_2'])
        ))
        self.linear1 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.hidden['encoder_2'], self.hidden['decoder_1']), 
            nn.ReLU(), 
            nn.Linear(self.hidden['decoder_1'], self.hidden['decoder_2']),
            nn.ReLU(),
            nn.Linear(self.hidden['decoder_2'], self.hidden['decoder_3']),
        ))
        self.linear2 = nn.Sequential(*modules)
        self.linear3 = nn.Sequential(*modules)
        self.linear4 = nn.Sequential(*modules)

    def dashun_model(self, eta, mu, sigma, t):
        tt = torch.full(mu.shape, t).to(self.device)
        x = (torch.log(tt)-mu)/(1+sigma)
        norm = Normal(loc=0.0, scale=1.0)
        inte = norm.cdf(x)
        return torch.exp(eta*inte) - 1

    def forward(self, c):
        # _, c = self.gru(imputed_embs) # [b, 5, d] -> [1, b, 50]
        # z = self.linear1(c.squeeze(0)) # [b, 50] -> [b, 10]

        z = self.linear1(c)
        eta = self.linear2(z) # [b, 10] -> [b, 1]
        mu = self.linear3(z)
        sigma = self.linear4(z)

        preds = [self.dashun_model(eta, mu, sigma, t) for t in range(1, 1+self.pred_year)] # list([b, 1])
        pred = torch.hstack(preds) # [b, 5]
        return pred


class FC(nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()
        self.pred_year = args.pred_year # 5
        self.device = args.device
        self.hidden = {"encoder_1":50,"encoder_2":64,
                           "decoder_1":32,"decoder_2":16,"decoder_3":5,
                           "rnn":50,"conditional_1":64}
        
        self.gru = nn.GRU(args.embed_dim, self.hidden['rnn'], batch_first=True)
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(args.hidden_channels, self.hidden['conditional_1']), 
            nn.ReLU(), 
            nn.Linear(self.hidden['conditional_1'], self.hidden['encoder_2'])
        ))
        self.linear1 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.hidden['encoder_2'], self.hidden['decoder_1']), 
            nn.ReLU(), 
            nn.Linear(self.hidden['decoder_1'], self.hidden['decoder_2']),
            nn.ReLU(),
            nn.Linear(self.hidden['decoder_2'], self.hidden['decoder_3']),
        ))
        self.linear2 = nn.Sequential(*modules)


    def forward(self, c):
        z = self.linear1(c) # 128 -> 64 -> 64
        pred = self.linear2(z) # 64->32->16->5
        return pred
    
    
class Seq2Seq(nn.Module):
    def __init__(self, args, dropout = 0.2):
        super().__init__()

        self.hidden = {"encoder_1":50,"encoder_2":10,
                           "decoder_1":20,"decoder_2":8,"decoder_3":1,
                           "rnn":50,"conditional_1":20}
        self.out_dim = 5
        self.dropout1 = nn.Dropout(dropout)
        self.encoder = nn.LSTM(args.embed_dim, self.hidden['rnn'], batch_first=True)#, dropout = dropout)

        self.embedding = nn.Embedding(self.out_dim, args.embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.decoder = nn.LSTM(args.embed_dim, self.hidden['rnn'], batch_first=True)
        self.linear1 = nn.Linear(self.hidden['rnn'], self.out_dim)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(self.hidden['encoder_1'], self.hidden['decoder_1']), 
            nn.ReLU(), 
            nn.Linear(self.hidden['decoder_1'], self.hidden['decoder_2']),
            nn.ReLU(),
            nn.Linear(self.hidden['decoder_2'], self.hidden['decoder_3']),
        ))
        self.linear = nn.Sequential(*modules)

    def forward(self, imputed_embs):
        encoder_outputs, (hidden, cell) = self.encoder(imputed_embs) # [b, 5, d] -> [1, b, 50]
        outputs, (hidden, cell) = self.decoder(encoder_outputs, (hidden, cell)) # [b, 5, d] -> [b, 5, 50]

        pred = self.linear(outputs) # [b, 5, 50] -> [b, 5, 1]
        return pred.squeeze(2)