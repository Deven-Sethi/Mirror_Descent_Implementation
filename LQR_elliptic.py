import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as multi_norm_sampler
import numpy as np
import itertools
from tqdm import tqdm
import NN_for_PDEs

class auto_differentiation_helper():
    def __init__(self, v):
        self.v = v # v = function to be differentiated

    
    def get_gradient(self, x):
        grad_v = torch.autograd.grad(self.v, 
                                     x, 
                                     grad_outputs=torch.ones_like(self.v), 
                                     create_graph=True, 
                                     retain_graph=True, 
                                     only_inputs=True)[0]
        return(grad_v)

    def get_laplacian(self, grad_v, x):
        hess_diag = []
        for d in range(x.shape[1]):
            v = grad_v[:,d].view(-1,1)
            grad2 = torch.autograd.grad(v, 
                                        x, 
                                        grad_outputs = torch.ones_like(v), 
                                        only_inputs=True, 
                                        create_graph=True, 
                                        retain_graph=True)[0]
            hess_diag.append(grad2[:,d].view(-1,1))    
        hess_diag = torch.cat(hess_diag,1)
        lap_v = hess_diag.sum(1, keepdim=True)
        return(lap_v)

def batch_vect_mat_vect(x, M):
    x_tMx = torch.sum((torch.matmul(x, M) * x), axis = 1)
    return(x_tMx)

def get_gradient(v, x):
    grad_v = torch.autograd.grad(v, x,grad_outputs=torch.ones_like(v), 
                                     create_graph=True, 
                                     retain_graph=True, 
                                     only_inputs=True)[0]
    return(grad_v)

def get_laplacian(grad_v, x):
        hess_diag = []
        for d in range(x.shape[1]):
            v = grad_v[:,d].view(-1,1)
            grad2 = torch.autograd.grad(v, x, grad_outputs = torch.ones_like(v), only_inputs=True, create_graph=True,retain_graph=True)[0]
            hess_diag.append(grad2[:,d].view(-1,1))    
        hess_diag = torch.cat(hess_diag,1)
        lap_v = hess_diag.sum(1, keepdim=True)
        return(lap_v)


class LQR_problem_reg():
    def __init__(self, A, B, sigma, M, R, N, P, Q, rho, tau):
        self.A = A.float() 
        self.B = B 
        self.sigma = sigma
        self.M = M.float()
        self.R = R 
        self.N = N 
        self.P = P 
        self.Q = Q
        self.rho = rho
        self.tau = tau


    
    def generate_from_state_space(self, N):
        '''
        generate from the uniform distribution over the state space
        which we take to be the ball of radius rad
        '''
        rand_rad =  6 * torch.sqrt(torch.rand(N))
        rand_ang =  2 * np.pi * torch.rand(N)
        x_cord = rand_rad * np.cos(rand_ang)
        y_cord = rand_rad * np.sin(rand_ang)
        return(torch.stack((x_cord,y_cord), 1))

    
    def learn_value_function(self, dimX, dimH, lr, num_of_epochs, num_samples, mu_samples):
        '''
        dimX = spatial dimension, dimH = hidden dimensions, lr = learning rate
        '''
        model_value_funct = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model_value_funct.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs) 
        A_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((mu_samples,))
        
        Q_A_samples = torch.sum(A_samples * self.Q, axis = 1)
        A_N_A = torch.sum(torch.matmul(A_samples, self.N.float()) * A_samples, axis = 1)
        a_TR = torch.matmul(A_samples, self.R.unsqueeze(0).float()).squeeze()

        learning_loss = [100 for i in range(30)]

        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            train_samples_x = self.generate_from_state_space(num_samples) 
            #train_samples_x = torch.tensor(np.random.uniform(4, 4, (num_samples, 2)))          
            x_batch = train_samples_x.float()
            x_batch.requires_grad = True

            train_v_x = model_value_funct(x_batch)
            if torch.isnan(torch.sum(train_v_x)).item():
                print('nan')
            grad_train_v_x = get_gradient(train_v_x, x_batch)
            lap_train_v_x = get_laplacian(grad_train_v_x, x_batch)

            x_TM_x = batch_vect_mat_vect(x_batch, self.M).reshape(num_samples,1)
            AX_trans = torch.matmul(self.A, x_batch.unsqueeze(-1)).reshape(num_samples,2)
            AX_trans_grad_v = torch.sum(AX_trans * grad_train_v_x, axis = 1).reshape(num_samples,1)
            Px = torch.sum(x_batch * self.P, axis = 1).reshape(num_samples,1)
            linear_PDE_part = 0.5 * self.sigma**2 * lap_train_v_x - self.rho * train_v_x + 0.5 * x_TM_x + AX_trans_grad_v + Px
            
            stretched_grad_v = (grad_train_v_x.repeat_interleave(mu_samples, dim = 0)).reshape(num_samples,mu_samples, dimX)
            a_v_grad = torch.sum(stretched_grad_v * A_samples, axis = 2)
            x_batch_Stretch = x_batch.repeat_interleave(mu_samples, 0).reshape(num_samples, mu_samples,  2)
            a_TR_x = torch.sum(a_TR * x_batch_Stretch, axis = 2)
                        
            non_linear_part = torch.log((1 / mu_samples ) * torch.sum(torch.exp(- ( a_v_grad + a_TR_x + Q_A_samples + 0.5 * A_N_A ) / self.tau),axis = 1)).reshape(num_samples,1)
            if torch.isnan(torch.sum(non_linear_part)).item():
                print('nan')

            pde = linear_PDE_part - self.tau * non_linear_part

            if torch.isnan(torch.sum(pde)).item():
                print('nan')
            target_functional = torch.zeros_like(train_v_x)
            MSE_functional = loss_fn(pde, target_functional)
            learning_loss.append(MSE_functional.item())


                  
            loss = MSE_functional
            loss.backward()
            optimizer.step()

            pbar.update(1)

            if np.sum(learning_loss[-30:]) / 20 <= 0.001:
                break
            
        return(model_value_funct, learning_loss)
    
 
    def next_step(self, old_step, old_actions,step_size, N):
        return(old_step + (torch.matmul(old_step, self.A) + torch.matmul(old_actions, self.B)) * step_size + self.sigma * np.sqrt(step_size) * torch.randn(N,2))

    
    def MC_value_function_approx(self, MC_samples, total_time_Steps, T , initial_cond, model, exact_val):
        # sample O times from mu
        Del_t = T / total_time_Steps
        Sigma = self.tau * torch.inverse((self.N.float() + self.tau * torch.eye(2).float())).expand(MC_samples, 2, 2)
        approx = 0
        X0 = initial_cond.repeat(MC_samples,1).float()
        X0.requires_grad = True 

        mu_samples = 4000
        A_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((mu_samples,))

        error = []
        #pbar = tqdm(total = total_time_Steps) 

        def get_gradient_no_grad(v, x):
            grad_v = torch.autograd.grad(v, x,grad_outputs=torch.ones_like(v))[0] 
                                            #create_graph=True, 
                                            #retain_graph=True, 
                                            #only_inputs=True)[0]
            return(grad_v)

        
        for i in range(total_time_Steps):
            v_X0 = model(X0)
            v_X0_grad = get_gradient_no_grad(v_X0, X0)
            #v_X0_grad_no_grad = v_X0_grad.detach()
            #v_X0_grad_no_grad.requires_grad = False

            mean_batch = -torch.matmul(Sigma/self.tau, (v_X0_grad + self.Q).reshape(MC_samples, 2, 1)).reshape(MC_samples,2)
            samples = multi_norm_sampler.MultivariateNormal(mean_batch, Sigma).sample()
            #samples = sampler.sample()

            #X0_no_grad = X0.detach()
            #X0_no_grad.requires_grad = False

            discounting = np.exp(-self.rho * i * Del_t)
            x_TM_x = torch.sum(torch.matmul(X0, self.M) * X0, axis = 1)
            Px = torch.matmul(X0,self.P.float()) 

            aDv = v_X0_grad.repeat(mu_samples,1).reshape(MC_samples,mu_samples,2)
            a_v_grad = torch.sum(aDv * A_samples, axis = 2)
            #a_v_grad = torch.sum(v_X0_grad_no_grad.repeat(mu_samples,1).reshape(MC_samples,mu_samples,2) * A_samples, axis = 2)
            Q_A_samples = torch.sum(A_samples * self.Q, axis = 1)
            A_N_A = torch.sum(torch.matmul(A_samples, self.N.float()) * A_samples, axis = 1)
            ex_Z_star_tau = a_v_grad + Q_A_samples + 0.5 * A_N_A
            int_mu_da = torch.log((1 / mu_samples ) * torch.sum(torch.exp(- ex_Z_star_tau / self.tau), axis = 1))
            
            
            integral = discounting * torch.sum(0.5 * x_TM_x + Px - self.tau * int_mu_da - torch.sum(v_X0_grad * samples, axis = 1)).item() * Del_t / total_time_Steps
            approx = approx + integral

            #del v_X0_grad_no_grad
            #del X0_no_grad

            error.append(abs(approx - exact_val))

            X1 = self.next_step(X0, samples, Del_t, MC_samples)
            X0 = X1

            if i % 20 == 0:
                print(approx)

            #pbar.update(1)
        return(approx, error)




