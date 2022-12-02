import pickle as pkl
import os
import torch
import numpy as np
import random 

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, SAGEConv
from torch.distributions.normal import Normal

from .gnn import GCN, GAT, SAGE


class NODE_NET(nn.Module):
    def __init__(self, args):
        super(NODE_NET, self).__init__()
        print(f'linkgnn = {args.linkgnn}')
        if args.linkgnn == 'gcn':
            self.gnn = GCN(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        if args.linkgnn == 'gat':
            self.gnn = GAT(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        else:
            self.gnn = SAGE(args.embed_dim, args.hidden_channels, args.hidden_channels, \
                args.num_layers, args.dropout).to(args.device)
        
        print('node_predictor', args.node_predictor)
        if args.node_predictor == 'fc':
            self.predictor = FcPredictor(args.hidden_channels, args.hidden_dims, \
                args.node_year, args.device).to(args.device)
            
        else:
            self.predictor = CitePredictor(args.hidden_channels, args.hidden_dims, \
                args.node_year, args.device).to(args.device)


class CitePredictor(nn.Module):
    def __init__(self, hidden_channels, hidden_dims = [20, 10, 20, 8], node_year=5, device=torch.device('cuda')):
        super(CitePredictor, self).__init__()
        self.node_year = node_year
        self.device = device
        
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_channels, hidden_dims[0]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[0], hidden_dims[1])
        ))
        self.linear1 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], 1),
        ))
        self.linear2 = nn.Sequential(*modules)
        self.linear3 = nn.Sequential(*modules)
        self.linear4 = nn.Sequential(*modules)
        
        
    def forward(self, c):
        z = self.linear1(c)
        eta = self.linear2(z) # [b, 10] -> [b, 1]
        mu = self.linear3(z)
        sigma = self.linear4(z)

        preds = [self.cites_model(eta, mu, sigma, t) for t in range(1, 1+self.node_year)] # list([b, 1])
        pred = torch.hstack(preds) # [b, 5]
        return pred
    
    
    def cites_model(self, eta, mu, sigma, t):
        tt = torch.full(mu.shape, t).to(self.device)
        x = (torch.log(tt)-mu)/(1+sigma)
        norm = Normal(loc=0.0, scale=1.0)
        inte = norm.cdf(x)
        return torch.exp(eta*inte) - 1



class FcPredictor(nn.Module):
    def __init__(self, hidden_channels, hidden_dims = [20, 10, 20, 8]):
        super(FcPredictor, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_channels, hidden_dims[0]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[0], hidden_dims[1])
        ))
        self.linear1 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], 1),
        ))
        self.linear2 = nn.Sequential(*modules)


    def forward(self, c):
        z = self.linear1(c) # 128 -> 64 -> 64
        pred = self.linear2(z) # 64->32->16->5
        return pred