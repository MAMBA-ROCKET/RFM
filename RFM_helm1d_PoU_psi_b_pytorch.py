# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
import matplotlib.pyplot as plt
import random
import pandas as pd

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
        nn.init.uniform_(m.weight, a = -8, b = 8)
        nn.init.uniform_(m.bias, a = -8, b = 8)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)

# network definition
class RFM_rep(nn.Module):
    def __init__(self, in_features, J_n, x_max, x_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features
        self.hidden_features = J_n # hidden features as the number of basis functions in every point
        self.J_n = J_n
        self.x_min = x_min
        self.x_max = x_max
        self.a = 2.0/(x_max - x_min) # 1/radius
        self.x_0 = (x_max + x_min)/2 # center point
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh()) 
        # nn.linear will generate Jn basis functions and they will pass into the next layer
        # this NN is learning the weight of every basis function I guess this makes sense

    # def forward(self,x):
    #     d = (x - self.x_min) / (self.x_max - self.x_min)
    #     #d = (x - self.x_0) * self.a
    #     d0 = (d <= -1/4)
    #     d1 = (d <= 1/4)  & (d > -1/4)
    #     d2 = (d <= 3/4)  & (d > 1/4)
    #     d3 = (d <= 5/4)  & (d > 3/4)
    #     d4 = (d > 5/4)
    #     y = self.a * (x - self.x_0)  # here y is the input of hidden layer, as the x variable
    #     y = self.hidden_layer(y) # The left hand side y is the local approximation of every point 
    #     y0 = 0
    #     y1 = y * (1 + torch.sin(2*np.pi*d) ) / 2
    #     y2 = y
    #     y3 = y * (1 - torch.sin(2*np.pi*d) ) / 2
    #     y4 = 0
    #     if self.x_min == 0:
    #         return(d0*y0+(d1+d2)*y2+d3*y3+d4*y4)
    #     elif self.x_max == interval_length:
    #         return(d0*y0+d1*y1+(d2+d3)*y2+d4*y4)
    #     else:
    #         return(d0*y0+d1*y1+d2*y2+d3*y3+d4*y4)
        

    def forward(self,x):
        d = (x - self.x_min) / (self.x_max - self.x_min)
        #d = (x - self.x_0) * self.a
        d0 = (d <= -1/4)
        d1 = (d <= 1/4)  & (d > -1/4)
        d2 = (d <= 3/4)  & (d > 1/4)
        d3 = (d <= 5/4)  & (d > 3/4)
        d4 = (d > 5/4)
        y = self.a * (x - self.x_0)  # here y is the input of hidden layer, as the x variable
        y = self.hidden_layer(y) # The left hand side y is the local approximation of every point 
        y0 = 0
        y1 = y * (1 + torch.sin(2*np.pi*d) ) / 2
        y2 = y
        y3 = y * (1 - torch.sin(2*np.pi*d) ) / 2
        y4 = 0
        if self.x_min == 0:
            return((d1+d2)*y2+d3*y3)
            #return(d0*y0+(d1+d2)*y2+d3*y3+d4*y4)
        elif self.x_max == interval_length:
            return(d1*y1+(d2+d3)*y2)
            #return(d0*y0+d1*y1+(d2+d3)*y2+d4*y4)
        else:
            return(d1*y1+d2*y2+d3*y3)
            #return(d0*y0+d1*y1+d2*y2+d3*y3+d4*y4)

# analytical solution parameters
AA = 1
aa = 2.0*np.pi
bb = 3.0*np.pi
interval_length = 8.
lamb = 4

def anal_u(x):
    return AA * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) + 2.0

def anal_dudx_1st(x):
    return AA * (bb*np.cos(bb*(x+0.05))*np.cos(aa*(x+0.05)) - aa*np.sin(bb*(x+0.05))*np.sin(aa*(x+0.05)))

def anal_dudx_2nd(x):
    return -AA*(aa*aa+bb*bb)*np.sin(bb*(x+0.05))*np.cos(aa*(x+0.05))\
           -2.0*AA*aa*bb*np.cos(bb*(x+0.05))*np.sin(aa*(x+0.05))

def Lu_f(pointss, lambda_ = 4):
    r = []
    for x in pointss:
        f = anal_dudx_2nd(x)
        r.append(f)
    return(np.array(r))


# define the local-networks and points in the corresponding regions
def pre_define(M_p,J_n,Q):
    models = []
    points = []
    for k in range(M_p):
        x_min = 8.0/M_p * k
        x_max = 8.0/M_p * (k+1) # This means chopping the interval into M_p pieces and deal with each piece seperately
        model = RFM_rep(in_features = 1, J_n = J_n, x_min = x_min, x_max = x_max) # this in_feature is the input of the first layer, as x?
        model = model.apply(weights_init)
        model = model.double()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        points.append(torch.tensor(np.linspace(x_min, x_max, Q+1),requires_grad=True).reshape([-1,1]))
        # Every interval has Q+1 points, and the first and last point are the boundary points
    #print('points:',points)
    #print('number of points:',len(points))
    return(models,points)




# calculate the matrix A,f in linear equations system 'Au=f'
def cal_matrix(models,points,M_p,J_n,Q):
    # matrix define (Aw=b)
    A_1 = np.zeros([M_p*(Q+1),M_p*J_n])
    A_2 = np.zeros([2,M_p*J_n])
    f = np.zeros([M_p*(Q+1) + 2, 1])
    
    for k in range(M_p):
        # forward and grad
        for m in range(M_p):
            out = models[m](points[k])
            values = out.detach().numpy()
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
            Lu = grads_2
            print(Lu.shape)
            # Lu = f condition
            A_1[k*(Q+1):(k + 1)*(Q+1), m*J_n:(m + 1)*J_n] = Lu[:Q+1,:]
            # boundary condition
            if k == 0 and m==k:
                A_2[0, :J_n] = values[0,:]
            elif k == M_p - 1 and m==k:
                A_2[1, -J_n:] = values[-1,:]
                
        true_f = Lu_f(points[k].detach().numpy(), lamb).reshape([(Q + 1),1])
        f[k*(Q+1):(k + 1)*(Q+1),: ] = true_f[:Q+1]
    A = np.concatenate((A_1,A_2),axis=0)
    f[M_p*(Q+1),:] = anal_u(0.)
    f[M_p*(Q+1)+1,:] = anal_u(8.)
    return(A,f)


# calculate the l^{inf}-norm and l^{2}-norm error for u,v,p
def test(models,M_p,J_n,Q,w,plot = True):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Q = int(1000/M_p)
    for k in range(M_p): # k is for domain decomposition
        points = torch.tensor(np.linspace(8.0/M_p * (k), 8.0/M_p * (k+1), test_Q+1),requires_grad=False).reshape([-1,1])
        #print('points:',points)
        out_total = None
        for m in range(M_p):
            out = models[m](points) # m is for model defined in every subdomain
            values = out.detach().numpy()
            if out_total is None:
                out_total = values
            else:
                out_total = np.concatenate((out_total,values),axis=1)
        true_value = anal_u(points.numpy()).reshape([-1,1])
        #print('out_total:',out_total)
        numerical_value = np.dot(np.array(out_total), w)
        #print('numerical_value:',numerical_value)
        true_values.extend(true_value)
        numerical_values.extend(numerical_value)
        epsilon.extend(true_value - numerical_value)
    true_values = np.array(true_values)

    numerical_values = np.array(numerical_values)
    # for i in range(len(numerical_values)):
    #     print(numerical_values[i])
    #print('Approximate solution:',numerical_values)
    #print('Exact solution:',true_values)
    # print(true_values.shape)
    print(np.max(np.abs(true_values - numerical_values)))
    epsilon = np.array(epsilon)
    epsilon = np.maximum(epsilon, -epsilon)
    print('********************* ERROR *********************')
    print('M_p=%s,J_n=%s,Q=%s'%(M_p,J_n,Q))
    print('L_inf=',epsilon.max(),'L_2=',math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))
        
    x = [(interval_length/M_p)*i / test_Q  for i in range(M_p*(test_Q+1))]
    if plot == True:
        plt.figure()
        plt.plot(x, true_values, label = "exact solution", color='black')
        plt.plot(x, numerical_values, label = "numerical solution", color='darkblue', linestyle='--')
        plt.legend()
        plt.title('exact solution')
        plt.savefig('./numerical_solution.pdf', dpi=100)
        
        plt.figure()
        plt.plot(x, epsilon, label = "absolute error", color='black')
        plt.legend()
        plt.title('RFM error, $\psi^2$, J_n=%s Q=%s'%(M_p*J_n,M_p*Q))
        #plt.savefig('./error_Ne=%sJ_n=%sQ=%s.pdf'%(M_p,J_n,Q), dpi=100)
    return(epsilon.max(),math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))


def main(M_p,J_n,Q,plot = True, moore = False):
    # prepare models and collocation pointss
    models,points = pre_define(M_p,J_n,Q)
    
    # matrix define (Aw=b)
    A,f = cal_matrix(models,points,M_p,J_n,Q)
    df = pd.DataFrame(A)
    df.to_excel("Lu_author.xlsx", index=False, header=False, engine='openpyxl')

    hessian = np.matmul(A.T,A)
    print('target value:',f)
    print('hessian',hessian)

    print('loss:',np.sum(f**2))
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]
        res = (np.dot(A,w) - f)
        print(w)
        print('residue2:',np.sum(res**2))
        print('residue:',np.max(np.abs(res)))
        #print('residue:',lstsq(A,f)[1])
    
    # test
    # w = np.ones_like(w)
    # for i,res in enumerate(w):
    #     print(res)
    #     print(i)
    print('number of coefficient:',len(w))
    return(test(models,M_p,J_n,Q,w,plot))



if __name__ == '__main__':
    set_seed(100)
    #M_p = 4 # the number of basis center points
    J_n = 50 # the number of basis functions per center points
    Q = 50 # the number of collocation pointss per basis functions support
    for M_p in [4]:
        main(M_p,J_n,Q,True,False)