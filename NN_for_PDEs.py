import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DGM_Layer(nn.Module):
    '''
    This code is copied from Marc's GitHub
    https://github.com/msabvid/Deep-PDE-Solvers/blob/master/lib/dgm.py
    '''
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
            

        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
            
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output

class Net_DGM(nn.Module):
    '''
    This code is copied from Marc's GitHub
    https://github.com/msabvid/Deep-PDE-Solvers/blob/master/lib/dgm.py
    '''
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,x):
        x_cat = torch.cat([x], 1)
        S1 = self.input_layer(x_cat)
        S2 = self.DGM1(x_cat, S1)
        S3 = self.DGM2(x_cat, S2)
        S4 = self.DGM3(x_cat, S3)
        output = self.output_layer(S4)
        return output
