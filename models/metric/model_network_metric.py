import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import LayerNorm, InstanceNorm1d
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer
import torch.nn as nn

torch.backends.cudnn.benchmark = True


def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B, ctx_dim=None, use_film=True):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 256 #512,256
        #input_size = 128
        #self.T=2

        self.B = B.T.to(device)
        print(B.shape)
        input_size = B.shape[0]
        #decoder

        self.scale = 10
        #self.actvn = torch.sin()#nn.Softplus(beta=self.scale)


        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU
        #self.ddact = torch.nn.Sigmoid()-torch.nn.Sigmoid()*torch.nn.Sigmoid()
        self.actout = Sigmoid_out()#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU

        self.nl1=2
        #self.nl2=2

        self.encoder = torch.nn.ModuleList()
        self.encoder_norm = InstanceNorm1d(h_size)#torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(252,h_size))

        for i in range(0,3*self.nl1):
            self.encoder.append(Linear(h_size, h_size)) 
        self.encoder.append(Linear(h_size, h_size)) 

        self.gate = torch.nn.ModuleList()
        for i in range(self.nl1):
            self.gate.append(Linear(1,1))

        self.pe_gate = torch.nn.ModuleList()
        self.pe_gate.append(Linear(h_size,h_size))
        self.pe_gate.append(Linear(h_size,h_size))

        # -------------------------
        # FiLM / Context conditioning
        # -------------------------
        self.use_film = use_film
        if ctx_dim is None:
            # ctx = [speed_mu(2), speed_var(2), normal_mu(2d), normal_var(2d)] => 4 + 4d
            ctx_dim = 4 + 4*self.dim
        self.ctx_dim = ctx_dim

        if self.use_film:
            hidden = 128
            self.ctx_net = nn.Sequential(
                nn.Linear(self.ctx_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2*h_size),
            )
            # Initialize FiLM to identity: gamma ~ 1, beta ~ 0
            nn.init.zeros_(self.ctx_net[-1].weight)
            nn.init.zeros_(self.ctx_net[-1].bias)

    def film(self, x, ctx):
        """
        x: (N, h_size)
        ctx: (N, ctx_dim) or (2N, ctx_dim) depending on how we batch
        Returns: FiLM-modulated x
        """
        if (ctx is None) or (not self.use_film):
            return x

        # Ensure ctx is not part of autograd graph w.r.t coords
        ctx = ctx.detach()

        gb = self.ctx_net(ctx)  # (N, 2*h_size)
        g_raw, b_raw = torch.chunk(gb, 2, dim=-1)

        # Safe parameterization:
        # gamma in (1-0.1, 1+0.1), beta in (-0.1, 0.1)
        gamma = 1.0 + 0.1 * torch.tanh(g_raw)
        beta  = 0.1 * torch.tanh(b_raw)

        return gamma * x + beta

    #'''
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            #stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            stdv = np.sqrt(2.0 / (m.weight.size(0)+m.weight.size(1)))
            #stdv = np.sqrt(6 / 128.) #/ self.T
            #m.weight.data.trunc_normal_(0, variance)
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stdv, a=-2.0*stdv, b=2.0*stdv)
            m.bias.data.fill_(0)
        
        for i in range(self.nl1):
            self.gate[i].weight.data.fill_(0)
            self.gate[i].bias.data.fill_(0)

        
   
    #'''
    def input_mapping(self, x):
        w = 2.*np.pi*self.B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)
    
    def lip_norm(self, w):
        #y = x@w.T+b
        absrowsum = torch.sqrt(torch.sum ( w**2 , dim =1)).detach()
        #print(absrowsum.shape)
        #scale = torch.clamp (1 / absrowsum ,max=1)#.squeeze()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        #print(w.shape)
        return w * scale.unsqueeze(1) #[: , None ]
    
    def out(self, coords, ctx=None):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))
        
        
        x = self.input_mapping(x)

        # Build ctx for the stacked batch.
        # Expected ctx shape: (N, ctx_dim) corresponding to pairs (qs,qg)
        # We replicate it for x0 and x1 so both endpoints get the same conditioning.
        if self.use_film and (ctx is not None):
            if ctx.dim() != 2:
                raise ValueError(f"ctx must be 2D (N, ctx_dim). Got shape {tuple(ctx.shape)}")
            if ctx.shape[0] != size:
                raise ValueError(f"ctx batch {ctx.shape[0]} != coords batch {size}")
            if ctx.shape[1] != self.ctx_dim:
                raise ValueError(f"ctx dim {ctx.shape[1]} != expected {self.ctx_dim}")
            ctx_stacked = torch.vstack((ctx, ctx))  # (2N, ctx_dim)
            x = self.film(x, ctx_stacked)

        w = self.pe_gate[0].weight
        b = self.pe_gate[0].bias
        w = self.lip_norm(w)
        u = torch.sin(x@w.T+b)

        w = self.pe_gate[1].weight
        b = self.pe_gate[1].bias
        w = self.lip_norm(w)
        v = torch.sin(x@w.T+b)

        for ii in range(0,self.nl1):
            #i0 = x
            x_tmp = x
            
            w = self.encoder[3*ii+1].weight
            b = self.encoder[3*ii+1].bias

            w = self.lip_norm(w)

            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.encoder[3*ii+2].weight
            b = self.encoder[3*ii+2].bias

            w = self.lip_norm(w)

            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.encoder[3*ii+3].weight
            b = self.encoder[3*ii+3].bias

            w = self.lip_norm(w)

            y = x@w.T+b

            weight = torch.sigmoid(0.1*self.gate[ii].weight)

            x  = (1-weight)*x_tmp+(weight)*torch.sin(y)
            #x  = u*torch.sin(y)+v*x_tmp
            #x  = (1-weight)*x_tmp+weight*(u*torch.sin(y)+v*(1-torch.sin(y)))
            
        
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias

        w = self.lip_norm(w)
        
        y = x@w.T+b

        y = self.encoder_norm(y)

        x0 = y[:size,...]
        x1 = y[size:,...]

        #OURS
        x = torch.sqrt((x0-x1)**2+1e-6)
        x = x.view(x.shape[0],-1,16)
        x = (torch.logsumexp(10*x, 2)-np.log(16))/10
        x = 0.2*(torch.sum(x,dim=1,keepdim=True))

        #L1
        # x = 0.01*torch.norm(x0-x1,p=1,dim=1).unsqueeze(1)

        
        
        return x, w, coords
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        output, coords = self.out(coords)
        return output, coords
