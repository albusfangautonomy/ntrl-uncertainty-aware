import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
#from models import model_network_one as model_network
import igl 
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

# Prior versions relied on ``torch_kdtree`` for nearest neighbour queries.
# The implementation no longer depends on that package.

torch.backends.cudnn.benchmark = True


class Function():
    def __init__(self, path, device, network, dim):

        # ======================= JSON Template =======================
        self.path = path
        self.device = device
        self.dim = dim

        self.network = network

        # Pass the JSON information
        #self.Params['Device'] = device

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        #input_file = "datasets/gibson/Cabin/mesh_z_up_scaled.off"
        #self.kdtree, self.v_obs, self.n_obs = self.pc_kdtree(input_file)

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit/15.0
        self.offset = self.margin/10.0 
        # self.networks = networks  # list of ensemble members

    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    def out_and_grad_ensemble(self, points):
        """
        Returns:
        tau_mean: (N,1)
        dtau_mean: (N,2d)
        dtau_var: (N,2d)   # elementwise variance across ensemble members (epistemic)
        Xp_ref: (N,2d)     # coordinate tensor that requires grad (from first member)
        """
        taus = []
        dtaus = []
        Xp_ref = None

        for m, net in enumerate(self.networks):
            tau_m, w_m, Xp_m = net.out(points)
            dtau_m = self.gradient(tau_m, Xp_m)

            taus.append(tau_m)
            dtaus.append(dtau_m)

            if Xp_ref is None:
                Xp_ref = Xp_m  # keep for downstream code (TD step etc.)

        tau_stack = torch.stack(taus, dim=0)      # (M,N,1)
        dtau_stack = torch.stack(dtaus, dim=0)    # (M,N,2d)

        tau_mean = tau_stack.mean(dim=0)          # (N,1)
        dtau_mean = dtau_stack.mean(dim=0)        # (N,2d)

        # unbiased=False is fine; we just need a stable moment estimate
        dtau_var = dtau_stack.var(dim=0, unbiased=False)  # (N,2d)

        return tau_mean, dtau_mean, dtau_var, Xp_ref

    # def Loss(self, points, Yobs, normal, beta, gamma, epoch):
    def Loss(self, points, Yobs, Yvar, normal, normal_var, beta, gamma, epoch):

        start=time.time()
        # tau, w, Xp = self.network.out(points)
        # -------- build FiLM context (N, ctx_dim) --------
        ctx = torch.cat([
            Yobs,       # (N, 2)
            Yvar,       # (N, 2)
            normal,     # (N, 2d)
            normal_var  # (N, 2d)
        ], dim=1)

        # -------- forward pass with context --------
        tau, w, Xp = self.network.out(points, ctx=ctx)
        
        dtau = self.gradient(tau, Xp)
        end=time.time()
        
        #print(end-start)

        start=time.time()
        
        
        #tau, dtau, Xp = self.network.out_grad(points)
        
        end=time.time()
        #print(end-start)
        #print(dtau)

        #print(end-start)
        #print('')
        #y-x
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        D = torch.norm(Xp[:,self.dim:]-Xp[:,:self.dim], p=2, dim =1)
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        
        
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)

        td_weight = 1e-3#np.clip(1e-2*(1-(epoch)/300.), a_min=1e-3, a_max=1e-2) #2e-3
        with torch.no_grad():

            length0 = (0.03)/(Yobs[:,0]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir0 = length0*(DT0*Yobs[:,0].unsqueeze(1)**2).clone().detach()  
            #Dir1 = 0.03*(DT1/S1.unsqueeze(1)).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0, ctx=ctx)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length0#*1/Yobs[:,0].unsqueeze(1)
            del Xp_new0, Dir0#Xp_new1

        tau_loss0 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        #(1.01-Yobs[:,0])*td_weight*
        #(1.4-Yobs[:,0])*

        with torch.no_grad():

            length1 = (0.03)/(Yobs[:,1]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir1 = length1*(DT1*Yobs[:,1].unsqueeze(1)**2).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            #Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0
            Xp_new0[:,self.dim:] = Xp_new0[:,self.dim:] - Dir1
            #Xp_new[:,self.dim:]+=0.04*DT1/S1

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0, ctx=ctx)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length1#*1/Yobs[:,1].unsqueeze(1)
            del Xp_new0, Dir1#Xp_new1

        tau_loss1 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        
        where_d0 = (tau[:,0] < length0.squeeze())
        where_d1 = (tau[:,0] < length1.squeeze())
        tau_loss0[where_d0] = 0 
        tau_loss1[where_d1] = 0 

        tau_loss = tau_loss0+tau_loss1

        # -------------------------------
        # Mahalanobis Eikonal constraint
        # -------------------------------

        eik_weight = 1e-2
        eps = 1e-6

        # Gradient magnitudes
        gradnorm0 = torch.sqrt(S0 + eps)
        gradnorm1 = torch.sqrt(S1 + eps)

        # Eikonal residuals: ||∇τ|| - 1 / Y
        eik_r0 = gradnorm0 - 1.0 / Yobs[:, 0]
        eik_r1 = gradnorm1 - 1.0 / Yobs[:, 1]

        # Variance via first-order propagation:
        # Var(1/Y) ≈ (1 / Y^4) * Var(Y)
        eik_var0 = (Yvar[:, 0] / (Yobs[:, 0]**4)).detach() + eps
        eik_var1 = (Yvar[:, 1] / (Yobs[:, 1]**4)).detach() + eps

        # Mahalanobis (Gaussian NLL, scalar)
        eik_loss0 = eik_r0**2 / eik_var0 + torch.log(eik_var0)
        eik_loss1 = eik_r1**2 / eik_var1 + torch.log(eik_var1)

        # Optional spatial localization (same philosophy as normal loss)
        boundary_w0 = (1.001 - Yobs[:, 0])
        boundary_w1 = (1.001 - Yobs[:, 1])

        eik_loss = eik_weight * (
            boundary_w0 * eik_loss0 +
            boundary_w1 * eik_loss1
        )


        # -------------------------------
        # Mahalanobis normal constraint
        # -------------------------------

        normal_weight = 1e-3
        eps = 1e-6

        normal0 = normal[:, :self.dim]
        normal1 = normal[:, self.dim:]

        normal_var0 = normal_var[:, :self.dim]
        normal_var1 = normal_var[:, self.dim:]

        Yvar0 = Yvar[:, 0].unsqueeze(1)
        Yvar1 = Yvar[:, 1].unsqueeze(1)

        # Residuals: r_i = Y_i * ∇τ + n_i
        r0 = Yobs[:, 0].unsqueeze(1) * DT0 + normal0
        r1 = Yobs[:, 1].unsqueeze(1) * DT1 + normal1

        # Diagonal covariance (uncertainty-aware weighting)
        # Low speed (near obstacle) → higher uncertainty
        sigma0_sq = (Yvar0 * (DT0**2) + normal_var0).detach() + eps
        sigma1_sq = (Yvar1 * (DT1**2) + normal_var1).detach() + eps



        # Mahalanobis distance (diagonal Σ)
        mah0 = torch.sum(r0**2 / sigma0_sq, dim=1)
        mah1 = torch.sum(r1**2 / sigma1_sq, dim=1)

        # Log-determinant term (Gaussian NLL)
        logdet0 = torch.sum(torch.log(sigma0_sq), dim=1)
        logdet1 = torch.sum(torch.log(sigma1_sq), dim=1)

        # Spatial Weighting (Optional)
        boundary_w0 = (1.001 - Yobs[:,0])
        boundary_w1 = (1.001 - Yobs[:,1])

        # Final normal loss
        # n_loss = normal_weight * (mah0 + mah1 + logdet0 + logdet1)
        n_loss = normal_weight * (
            boundary_w0 * (mah0 + logdet0) +
            boundary_w1 * (mah1 + logdet1)
        )
        T = tau[:,0]
        #print(n_loss0)
        #T_num = T.clone.detach()+weight
        #
        #where = T>1
        #loss_td, loss_mpc = self.MPPIPlanner(Xp[where])
        #loss0 +
        #print(T)
        para = 0.5#max(0.5-epoch*0.001,0.5)
        #
        loss_n = (torch.sum((eik_loss+n_loss +tau_loss)*torch.exp(-0.5*T)))/Yobs.shape[0]#*torch.exp(-para*T)
        
        loss = beta*loss_n #+ 1e-4*(reg_tau)
        
        return loss, loss_n, eik_loss

    def TravelTimes(self, Xp):
     
        tau, w, coords = self.network.out(Xp)        

        TT = tau[:,0] #* torch.sqrt(T0)
            
        return TT

    def Speed(self, Xp):

   

        Xp = Xp.to(torch.device(self.device))

        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.einsum('ij,ij->i', D, D)

        #DT0 = dtau[:,self.dim:]
        DT0 = dtau[:,:self.dim]

        #T3    = tau[:,0]**2



        #TT = LogTau * torch.sqrt(T0)

        #print(tau.shape)

        #print(T02.shape)
        #T1    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        #T2    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        
        
        S = torch.einsum('ij,ij->i', DT0, DT0)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau#, T0#, T1, T2, T3
        return Ypred
    
    def Gradient(self, Xp):
        #Xp = Xp.to(torch.device(self.device))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)
        #print(Xp.shape)
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).view(-1,1)

        #A = T0*dtau[:,:self.dim]
        #B = tau/T0*D
        

        Ypred0 = -dtau[:,:self.dim]#-A+B
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0,dim=1).view(-1,1)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -dtau[:,self.dim:]#-A-B
        Spred1 = torch.norm(Ypred1,dim=1).view(-1,1)

        Ypred1 = 1/Spred1**2*Ypred1

        #print(Ypred0.shape)
        #print(Ypred1.shape)
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    
    def plot(self,epoch,total_train_loss,alpha,source):
        limit = 0.5#0.5
        xmin     = [-limit,-limit]
        xmax     = [limit,limit]
        spacing=limit/40.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))
        #dims_n = np.setdiff1d([0,1,2],dims)[0]

        size=81

        spacing = 2*limit/(size-1)#2.0#0.4
        X,Y  = np.meshgrid( np.arange(-limit,limit+0.1*spacing,spacing),
                            np.arange(-limit,limit+0.1*spacing,spacing))
        I,J  = np.meshgrid( np.arange(0,size,1),
                            np.arange(0,size,1))
        #print(X)
        #print(X.shape)
        #print(I.shape)
        V = np.zeros((size*size,3))
        V[:,0]=X.flatten()
        V[:,1]=Y.flatten()

        IDX = I*size+J

        F = np.zeros((2*(size-1)*(size-1),3))
        F[:(size-1)*(size-1),0]=IDX[:size-1,:size-1].flatten()
        F[:(size-1)*(size-1),1]=IDX[1:,:size-1].flatten()
        F[:(size-1)*(size-1),2]=IDX[:size-1,1:].flatten()
        F[(size-1)*(size-1):,0]=IDX[1:,1:].flatten()
        F[(size-1)*(size-1):,2]=IDX[1:,:size-1].flatten()
        F[(size-1)*(size-1):,1]=IDX[:size-1,1:].flatten()

        F=F.astype(int)

        Xsrc = source#[0]*self.dim
        #Xsrc[0] = -0.2
        #Xsrc[1] = -0.2
        #Xsrc[2] = -0.02
        #print(self.dim)
        #Xsrc=[-1.2, 0.4-0.5*np.pi, 1.4, 0.2-0.5*np.pi,-0.5,0.9]
        #-1.3, 0.6, 1.1, 0.2,-1.5,0.9
        #scale = np.pi/0.5
        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        #tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        #TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')
        plt.close(fig)
        #print(TT.flatten().shape)
        V[:,2] = TT.flatten()#0.2*
        igl.write_triangle_mesh(self.path+"/plots_TT"+str(epoch)+".obj", V, F) 


        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,0]  = X.flatten()
        XP[:,1]  = Y.flatten()
        XP[:,2] = -0.1
        XP = Variable(Tensor(XP)).to(self.device)

        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        #tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S  = ss.to('cpu').data.numpy().reshape(X.shape)
        #TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,S,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,10,0.02), cmap='gist_heat', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path+"/tauplots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)

        V[:,2] = TT.flatten()
        igl.write_triangle_mesh(self.path+"/plots_TAU"+str(epoch)+".obj", V, F) 


         
