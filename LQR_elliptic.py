import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as multi_norm_sampler
import numpy as np
import itertools
from tqdm import tqdm
import NN_for_PDEs

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

def vec_trans_mat_vec(x,M):
        '''
        x = N x 2 tensor, M = 2x2 matrix, we want the Nx1 tesnor given by x^{transpose} M x
        '''
        return(torch.sum(torch.matmul(x , M) * x, axis = 1))

class LQR_problem_reg():
    def __init__(self, A, B, sigma, M, N, rho, tau):
        self.A = A.float() 
        self.B = B 
        self.sigma = sigma
        self.M = M.float()
        #self.R = R 
        self.N = N 
        #self.P = P 
        #self.Q = Q
        self.rho = rho
        self.tau = tau
    
    def learn_value_function(self, dimX, dimH, lr, num_of_epochs, num_samples, mu_samples):
        ''' 

        this seems to be learning well.
        dimX = spatial dimension, dimH = hidden dimensions, lr = learning rate
        '''
        model_value_funct = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model_value_funct.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs) 
        A_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((mu_samples,))
        
        #Q_A_samples = torch.sum(A_samples * self.Q, axis = 1)
        A_N_A = torch.sum(torch.matmul(A_samples, self.N.float()) * A_samples, axis = 1)
        #a_TR = torch.matmul(A_samples, self.R.unsqueeze(0).float()).squeeze()

        learning_loss = [100 for i in range(30)]

        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            #train_samples_x = self.generate_from_state_space(num_samples) 
            train_samples_x = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((num_samples,))        
            x_batch = train_samples_x.float()
            x_batch.requires_grad = True

            train_v_x = model_value_funct(x_batch)
            grad_train_v_x = get_gradient(train_v_x, x_batch)
            lap_train_v_x = get_laplacian(grad_train_v_x, x_batch)

            x_TM_x = batch_vect_mat_vect(x_batch, self.M).reshape(num_samples,1)
            AX_trans = torch.matmul(self.A, x_batch.unsqueeze(-1)).reshape(num_samples,2)
            AX_trans_grad_v = torch.sum(AX_trans * grad_train_v_x, axis = 1).reshape(num_samples,1)
            linear_PDE_part = 0.5 * self.sigma**2 * lap_train_v_x - self.rho * train_v_x + 0.5 * x_TM_x + AX_trans_grad_v# + Px
            
            stretched_grad_v = (grad_train_v_x.repeat_interleave(mu_samples, dim = 0)).reshape(num_samples,mu_samples, dimX)
            a_v_grad = torch.sum(stretched_grad_v * A_samples, axis = 2)
            x_batch_Stretch = x_batch.repeat_interleave(mu_samples, 0).reshape(num_samples, mu_samples,  2)
            #a_TR_x = torch.sum(a_TR * x_batch_Stretch, axis = 2)
                        
            #non_linear_part = torch.log((1 / mu_samples ) * torch.sum(torch.exp(- ( a_v_grad + a_TR_x + Q_A_samples + 0.5 * A_N_A ) / self.tau),axis = 1)).reshape(num_samples,1)
            non_linear_part = torch.log((1 / mu_samples ) * torch.sum(torch.exp(- ( a_v_grad + 0.5 * A_N_A ) / self.tau),axis = 1)).reshape(num_samples,1)
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

            if np.sum(learning_loss[-30:]) / 25 <= 0.001:
                break
            
        return(model_value_funct, learning_loss)


    def learn_cost_for_control_mu(self, dimX, dimH, lr, num_of_epochs, num_samples, mu_samples):
        '''
        dimX = spatial dimension, dimH = hidden dimensions, lr = learning rate
        '''
        model_value_funct = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model_value_funct.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs) 
        
        A_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((mu_samples,))
        
        
        A_N_A = torch.sum(torch.matmul(A_samples, self.N.float()) * A_samples, axis = 1)
        A_NA_sum = torch.sum(A_N_A).item()
        aB_T = torch.matmul(A_samples, self.B.unsqueeze(0).float()).squeeze()
        learning_loss = [100 for i in range(30)]

        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            train_samples_x = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((num_samples,))
            #train_samples_x = self.generate_from_state_space(num_samples) 
            #train_samples_x = torch.tensor(np.random.uniform(4, 4, (num_samples, 2)))          
            x_batch = train_samples_x.float()
            x_batch.requires_grad = True

            train_v_x = model_value_funct(x_batch)
            grad_train_v_x = get_gradient(train_v_x, x_batch)
            lap_train_v_x = get_laplacian(grad_train_v_x, x_batch)

            x_TM_x = batch_vect_mat_vect(x_batch, self.M).reshape(num_samples,1)
            AX_trans = torch.matmul(self.A, x_batch.unsqueeze(-1)).reshape(num_samples,2)
            AX_trans_grad_v = torch.sum(AX_trans * grad_train_v_x, axis = 1).reshape(num_samples,1)

            stretched_grad_v = (grad_train_v_x.repeat_interleave(mu_samples, dim = 0)).reshape(num_samples,mu_samples, dimX)
            a_v_grad_sum = torch.sum(torch.sum(stretched_grad_v * aB_T, axis = 2), axis = 1)
            int_mu = (1 / mu_samples) * ( A_NA_sum + a_v_grad_sum )

            linear_PDE_part = 0.5 * (self.sigma**2)* lap_train_v_x - self.rho * train_v_x + 0.5 * x_TM_x + AX_trans_grad_v + int_mu.reshape(num_samples,1)
            
            pde = linear_PDE_part
            target_functional = torch.zeros_like(train_v_x)
            MSE_functional = loss_fn(pde, target_functional)
            learning_loss.append(MSE_functional.item())
             
            loss = MSE_functional
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if np.sum(learning_loss[-30:]) / 20 <= 0.0008:
                break
            
        return(model_value_funct, learning_loss)
        

        
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

    
 
    def next_step(self, old_step, old_actions,step_size, N):
        return(old_step + (torch.matmul(old_step, self.A) + torch.matmul(old_actions, self.B)) * step_size + self.sigma * np.sqrt(step_size) * torch.randn(N,2))



    #def on_policy_bellamn_for_policy_mu(self, dimX, dimH, lr, num_epochs,  num_samples):


    def learn_on_policy_bellman_for_policy_mu(self, dimX, dimH, lr, num_epochs,  num_samples):
        '''
        num_samples
        '''
        model = NN_for_PDEs.Net_DGM(dimX, dimH)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_epochs) 

        #def integrate_wrt_pi(grad, a_batch, num_samples, N, mu_samples):
        #    grad_stretch = grad.repeat_interleave(num_mu_samples, 0).reshape(num_samples,mu_samples,2)
        #    Ba_Dv = torch.sum(grad_stretch*a_batch, axis = -1)
        #    a_N_a = vec_trans_mat_vec(a_batch, self.N)
        #    return(torch.sum(Ba_Dv + 0.5 * a_N_a, axis = 1).reshape(num_samples,1))
        learning_loss = []
        A_T = self.A.T    
        
        #num_mu_samples = 4000
        #mu_sample = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((num_mu_samples,)) # this will be different for general control


        for epoch in range(num_epochs):
            optimizer.zero_grad
            spatial_samples  =  multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((num_samples,)) # will be here for arbitrary control
            
            #spatial_samples = self.generate_from_state_space(num_samples)
            spatial_samples.requires_grad = True
            
            policy_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((num_samples,)) # this will be different for general control
            
            v_x = model(spatial_samples)
            v_x_grad = get_gradient(v_x, spatial_samples)
            v_x_lap = get_laplacian(v_x_grad, spatial_samples)

            #xMx = vec_trans_mat_vec(spatial_samples, self.M)
            #Ax_T = 

            #pde = v_x_lap + v_x_grad - self.rho * v_x #+ 0.5 * xMx + AxDv

            x_TM_x = batch_vect_mat_vect(spatial_samples, self.M).reshape(num_samples,1)
            AX_trans = torch.matmul(self.A, spatial_samples.unsqueeze(-1)).reshape(num_samples,2)
            AX_trans_grad_v = torch.sum(AX_trans * v_x_grad, axis = 1).reshape(num_samples,1)
            Px = torch.sum(spatial_samples * self.P, axis = 1).reshape(num_samples,1)

            linear_PDE_part = 0.5 * self.sigma**2 * v_x_lap - self.rho * v_x + 0.5 * x_TM_x + AX_trans_grad_v + Px

            target_functional = torch.zeros_like(v_x)
            MSE_functional = loss_fn(linear_PDE_part, target_functional)
            learning_loss.append(MSE_functional.item())

            #boundary_samples = self.generate_from_boundary_state_space(num_samples)
            #v_x_bdry = model(boundary_samples)
            #boundary_values = torch.ones_like(v_x_bdry) * 25
            #MSE_bdry = loss_fn(v_x_bdry, boundary_values)
            
            loss = MSE_functional #+ MSE_bdry
            
            loss.backward()
            optimizer.step()
            pbar.update(1)

            if (epoch+1) % 20 == 0:
                print(epoch, np.mean(learning_loss[-10:]))




            '''
            v_x = model(spatial_samples)
            v_x_grad = get_gradient(v_x, spatial_samples)
            v_x_lap = get_laplacian(v_x_grad, spatial_samples)
            #int_pi = ( 1 / num_mu_samples) * integrate_wrt_pi(v_x_grad, mu_sample, num_samples, self.N, num_mu_samples)   # \int_{R^2} (Ba+a^TNa / 2)
            #int_pi = ( 1 / num_mu_samples) * integrate_wrt_pi(v_x_grad, mu_sample, num_samples, self.N, num_mu_samples)
            
            #x_TMx = vec_trans_mat_vec(spatial_samples, self.M).reshape(num_samples,1)
            #Ax_Dv = torch.sum(torch.matmul(spatial_samples, A_T) * v_x_grad, axis = 1).reshape(num_samples, 1)
            #pde = 0.5*self.sigma*self.sigma*v_x_lap - self.rho * v_x + 0.5 * x_TMx + Ax_Dv + int_pi
            pde = 0.5*self.sigma*self.sigma*v_x_lap - self.rho * v_x
            
            #boundary_samples = self.generate_from_boundary_state_space(num_samples)
            #boundary = 
            
            
            target_functional = torch.zeros_like(v_x)
            MSE_functional = loss_fn(pde, target_functional)
            learning_loss.append(MSE_functional.item())
            
            loss = MSE_functional
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if (epoch+1) % 20 == 0:
                print(epoch, np.mean(learning_loss[-10:]))
            '''
        return(model,learning_loss)

    
    def generate_from_state_space(self, N):
        '''
        generate from the uniform distribution over the state space
        which we take to be the ball of radius rad
        '''
        rand_rad =  5 * torch.sqrt(torch.rand(N))
        rand_ang =  2 * np.pi * torch.rand(N)
        x_cord = rand_rad * np.cos(rand_ang)
        y_cord = rand_rad * np.sin(rand_ang)
        return(torch.stack((x_cord,y_cord), 1))
    
    def generate_from_boundary_state_space(self, N):
        '''
        generate from the uniform distribution over the state space
        which we take to be the ball of radius rad
        '''
        rand_ang =  2 * np.pi * torch.rand(N)
        x_cord = 5 * np.cos(rand_ang)
        y_cord = 5 * np.sin(rand_ang)
        return(torch.stack((x_cord,y_cord), 1))

    


    




