# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
import matplotlib.pyplot as plt
import random

# fix random seed
torch.set_default_dtype(torch.float64)
def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True

# random initialization for parameters in FC layer
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -R_m, b = R_m)
        nn.init.uniform_(m.bias, a = -R_m, b = R_m)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)

# network definition
class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features
        self.hidden_features = J_n
        self.J_n = J_n
        self.x_min = x_min
        self.x_max = x_max
        self.a = 2.0/(x_max - x_min)
        self.x_0 = (x_max + x_min)/2
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())

    def forward(self,x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        #x = torch.sin(x)
        return x


# analytical solution parameters
AA = 1
aa = 2.0*np.pi
bb = 3.0*np.pi
interval_length = 8.
lamb = 4

def anal_u(x):
    return AA * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) + 2.0

def anal_dudx_2nd(x):
    return -AA*(aa*aa+bb*bb)*np.sin(bb*(x+0.05))*np.cos(aa*(x+0.05))\
           -2.0*AA*aa*bb*np.cos(bb*(x+0.05))*np.sin(aa*(x+0.05))

def Lu_f(pointss, lambda_ = 4):
    r = []
    for x in pointss:
        f = anal_dudx_2nd(x) - lambda_*anal_u(x)
        r.append(f)
    return(np.array(r))


# define the local-networks and points in the corresponding regions
def pre_define(J_n_p,J_n,Q):
    models = []
    points = []
    for k in range(J_n_p):
        x_min = 8.0/J_n_p * k
        x_max = 8.0/J_n_p * (k+1)
        model = RFM_rep(in_features = 1, J_n = J_n, x_min = x_min, x_max = x_max)
        model = model.apply(weights_init)
        model = model.double()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        points.append(torch.tensor(np.linspace(x_min, x_max, Q+1),requires_grad=True).reshape([-1,1]))
    return(models,points)


# calculate the matrix A,f in linear equations system 'Au=f'
def cal_matrix(models,points,M_p,J_n,Q,lamb):
    # matrix define (Aw=b)
    A_1 = np.zeros([M_p*Q,M_p*J_n])
    A_2 = np.zeros([2,M_p*J_n])
    A_3 = np.zeros([M_p-1, M_p*J_n])
    A_4 = np.zeros([M_p-1, M_p*J_n])
    f = np.zeros([M_p*Q + 2*(M_p - 1) + 2, 1])
    
    for k in range(M_p):
        # forward and grad
        out = models[k](points[k])
        values = out.detach().numpy()
        value_l = values[0,:]
        value_r = values[-1,:]
        grads = []
        grads_2 = []
        for i in range(J_n):
            g_1 = torch.autograd.grad(outputs=out[:,i], inputs=points[k],
                                  grad_outputs=torch.ones_like(out[:,i]),
                                  create_graph = True, retain_graph = True)[0]
            grads.append(g_1.squeeze().detach().numpy())
            g_2 = torch.autograd.grad(outputs=g_1[:,0], inputs=points[k],
                                  grad_outputs=torch.ones_like(out[:,i]),
                                  create_graph = False, retain_graph = True)[0]
            grads_2.append(g_2.squeeze().detach().numpy())
        grads = np.array(grads).T
        grads_2 = np.array(grads_2).T
        grad_l = grads[0,:]
        grad_r = grads[-1,:]
        Lu = grads_2 - lamb * values
        # Lu = f condition
        A_1[k*Q:(k + 1)*Q, k*J_n:(k + 1)*J_n] = Lu[:Q,:]
        
        true_f = Lu_f(points[k].detach().numpy(), lamb).reshape([(Q + 1),1])
        f[k*Q:(k + 1)*Q,: ] = true_f[:Q]
        if M_p > 1:
            # boundary condition and continuity condition
            if k == 0 :
                A_2[0, :J_n] = value_l
                A_3[0, :J_n] = -value_r
                A_4[0, :J_n] = -grad_r
            elif k == M_p - 1:
                A_2[1, -J_n:] = value_r
                A_3[M_p - 2, -J_n:] = value_l
                A_4[M_p - 2, -J_n:] = grad_l
            else:
                A_3[k-1,k*J_n:(k + 1)*J_n] = value_l
                A_4[k-1,k*J_n:(k + 1)*J_n] = grad_l
                A_3[k,k*J_n:(k + 1)*J_n] = -value_r
                A_4[k,k*J_n:(k + 1)*J_n] = -grad_r
        else:
            A_2[0, :J_n] = value_l
            A_2[1, -J_n:] = value_r
    if M_p > 1:
        A = np.concatenate((A_1,A_2,A_3,A_4),axis=0)
        print("debug")
    else:
        A = np.concatenate((A_1,A_2),axis=0)
    f[M_p*Q,:] = anal_u(0.)
    f[M_p*Q+1,:] = anal_u(8.)
    #print(f.shape)
    return(A,f)


# calculate the l^{inf}-norm and l^{2}-norm error for u
def test(models,M_p,J_n,Q,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Q = 2*Q
    for k in range(M_p):
        points = torch.tensor(np.linspace(8.0/M_p * (k), 8.0/M_p * (k+1), test_Q+1),requires_grad=False).reshape([-1,1])
        out = models[k](points)
        values = out.detach().numpy()
        true_value = anal_u(points.numpy()).reshape([-1,1])
        numerical_value = np.dot(values, w[k*J_n:(k+1)*J_n,:])
        true_values.extend(true_value)
        numerical_values.extend(numerical_value)
        epsilon.extend(true_value - numerical_value)
    true_values = np.array(true_values)
    numerical_values = np.array(numerical_values)
    epsilon = np.array(epsilon)
    epsilon = np.maximum(epsilon, -epsilon)
    print('********************* ERROR *********************')
    print('R_m=%s,M_p=%s,J_n=%s,Q=%s'%(R_m,M_p,J_n,Q))
    print('L_inf=',epsilon.max(),'L_2=',math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))
        
    x = [(interval_length/M_p)*i / test_Q  for i in range(M_p*(test_Q+1))]
    if plot == True:
        plt.figure()
        plt.plot(x, true_values, label = "true value", color='orange')
        plt.plot(x, numerical_values, label = "numerical solution", color='darkblue', linestyle='--')
        plt.legend()
        plt.title('local ELJ_n, $M_p$=%s J_n=%s Q=%s'%(M_p,J_n,Q))
        #plt.savefig('./resultNe=%sJ_n=%sQ=%s.pdf'%(M_p,J_n,Q), dpi=100)
        
        plt.figure()
        plt.plot(x, epsilon, label = "error", color='darkblue')
        plt.legend()
        plt.title('local ELJ_n error, $M_p$=%s J_n=%s Q=%s'%(M_p,J_n,Q))
        #plt.savefig('./result/resultNe=%sJ_n=%sQ=%serror.pdf'%(M_p,J_n,Q), dpi=100)
    return(epsilon.max(),math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))


def main(M_p,J_n,Q,lamb, plot = False, moore = False):
    # prepare models and collocation pointss
    models,points = pre_define(M_p,J_n,Q)
    
    # matrix define (Aw=b)
    A,f = cal_matrix(models,points,M_p,J_n,Q,lamb)
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]
    print(np.mean((np.dot(A,w)-f)**2))
    # test
    return(test(models,M_p,J_n,Q,w,plot))



if __name__ == '__main__':
    #set_seed(400)
    lamb = 4
    M_p = 4 # the number of basis center points
    J_n = 50 # the number of basis functions per center points
    #Q = 50 # the number of collocation pointss per basis functions support
    R_m = 3
    for Q in range(20,120,20):
        main(M_p,J_n,Q,lamb,True)