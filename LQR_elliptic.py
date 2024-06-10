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

def mean_rel_error(tensor_1,tensor_2):
        mre = abs(tensor_1-tensor_2) / abs(tensor_1)
        return(mre)

class LQR_problem_reg():
    def __init__(self, A, B, sigma, M, N, rho, tau, mu):
        self.A = A.float() 
        self.B = B 
        self.sigma = sigma
        self.M = M.float()
        self.N = N
        self.rho = rho
        self.tau = tau
        self.mu = mu
    
    def learn_uncontrolled_problem(self, dimX, dimH, lr, num_of_epochs, num_samples):
        model = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs)

        learning_loss = [100 for i in range(30)]

        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            #x_samples = self.generate_from_state_space(num_samples)
            x_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((num_samples,))
            x_samples.requires_grad = True
            v = model(x_samples)
            Dv = get_gradient(v, x_samples)
            Lap_v = get_laplacian(Dv, x_samples)
            drift = torch.matmul(x_samples, self.A)
            forcing = torch.sum(x_samples * x_samples, axis = 1).reshape(num_samples, 1)
            b_Dv = torch.sum(Dv * drift, axis = 1).reshape(num_samples, 1)
            pde = 0.5 * self.sigma**2 * Lap_v - self.rho * v + b_Dv + 0.5 * forcing

            target_functional = torch.zeros_like(v)
            MSE_functional = loss_fn(pde, target_functional)
            learning_loss.append(MSE_functional.item())

            #boundary_samples = self.generate_from_boundary_state_space(num_samples)
            #boundary_estimates = model(boundary_samples)
            #exact_boundary = torch.ones_like(boundary_estimates) * 25
            #MSE_boundary = loss_fn(boundary_estimates, exact_boundary) 
            
            loss = MSE_functional #+ MSE_boundary
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if epoch%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}".format(epoch, num_of_epochs, MSE_functional.item()))
                #pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}\t MSE boundary: {:.4f}".format(epoch, num_of_epochs, MSE_functional.item(), MSE_boundary.item()))
            if np.sum(learning_loss[-30:]) / 15 <= 0.0001:
                break
        return(model, learning_loss[30:])
    
    def MC_approx_no_control(self, x, MC, steps, dt):
        integral = 0
        X0 = x.repeat(MC,1)
        stopped_times = []
        for i in range(1,steps):
            discount = np.exp(-self.rho * i * dt)
            xMx_2 = 0.5 * torch.sum(X0 * X0, axis =1)
            integral = integral + discount * (dt/MC) * torch.sum(xMx_2, axis = 0)
            b_X0 = torch.matmul(X0, self.A)
            X1 = X0 + dt * b_X0 + self.sigma * np.sqrt(dt) * torch.randn(MC,2)
            X0 = X1
        return(integral.item())

    def learn_cost_funct_control_mu(self, dimX, dimH, lr, num_of_epochs, num_samples, num_mu_samples):
        model = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000, 1800],gamma=0.1)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs)
        learning_loss = [100 for i in range(30)]
        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            #x_samples = self.generate_from_state_space(num_samples)
            x_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((num_samples,))
            x_samples.requires_grad = True
            v = model(x_samples)
            Dv = get_gradient(v, x_samples)
            Lap_v = get_laplacian(Dv, x_samples)
            drift = torch.matmul(x_samples, self.A)
            forcing = torch.sum(x_samples * x_samples, axis = 1).reshape(num_samples, 1)
            b_Dv = torch.sum(Dv * drift, axis = 1).reshape(num_samples, 1)

            mu_samples = self.mu.sample((num_mu_samples,))
            aB = torch.matmul(mu_samples, self.B)
            stretch_Dv= Dv.repeat_interleave(num_mu_samples,0).reshape(num_samples,num_mu_samples,2)
            aNa = 0.5 * torch.sum(torch.matmul(mu_samples, self.N) * mu_samples, axis = 1)
            int_mu = (1/num_mu_samples) * torch.sum(torch.sum(stretch_Dv * aB.unsqueeze(0), axis = 2) + aNa, axis = 1).reshape(num_samples,1)
    
            pde = 0.5 * self.sigma**2 * Lap_v - self.rho * v + b_Dv + 0.5 * forcing + int_mu
 
            target_functional = torch.zeros_like(v)
            MSE_functional = loss_fn(pde, target_functional)
            learning_loss.append(MSE_functional.item())

            #boundary_samples = self.generate_from_boundary_state_space(num_samples)
            #boundary_estimates = model(boundary_samples)
            #exact_boundary = torch.ones_like(boundary_estimates) * 25
            #MSE_boundary = loss_fn(boundary_estimates, exact_boundary) 
            
            loss = MSE_functional #+ MSE_boundary
            loss.backward()
            optimizer.step()
            scheduler.step()
            #pbar.update(1)
            if epoch%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}".format(epoch, num_of_epochs, MSE_functional.item()))
                #pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}\t MSE boundary: {:.4f}".format(epoch, num_of_epochs, MSE_functional.item(), MSE_boundary.item()))
            if np.sum(learning_loss[-30:]) / 30 <= 0.001:
                break
        return(model, learning_loss[30:])

    def MC_approx_control_mu(self, x, MC, steps, dt):
        integral = 0
        X0 = x.repeat(MC,1)
        #A_samples_from_X0 = multi_norm_sampler.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((MC,))
        for i in range(1,steps):
            A_samples_from_X0 = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 0.8 * torch.eye(2)).sample((MC,))
            discount = np.exp(-self.rho * i * dt)
            xMx_2 = 0.5 * torch.sum(X0 * X0, axis =1)
            aNa_2 = 0.5 * torch.sum(torch.matmul(A_samples_from_X0, self.N) * A_samples_from_X0, axis = 1)
            integral = integral + discount * (dt/MC) * torch.sum(xMx_2 + aNa_2, axis = 0)
            b_X0 = torch.matmul(X0, self.A) + torch.matmul(A_samples_from_X0, self.B) 
            X1 = X0 + dt * b_X0 + self.sigma * np.sqrt(dt) * torch.randn(MC,2)
            X0 = X1
        return(integral.item())
    
    def learn_value_function(self, dimX, dimH, lr, num_of_epochs, num_samples, num_mu_samples):
        model = NN_for_PDEs.Net_DGM(dim_x = dimX, dim_S = dimH, activation='Tanh')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        #                                                 milestones=[100, 500, 1000, 1800, 2500],
        #                                                 gamma=0.1)
        loss_fn = nn.MSELoss() 
        pbar = tqdm(total = num_of_epochs)
        learning_loss = []
        learning_loss_against_exact = []
        for epoch in range(num_of_epochs):
            optimizer.zero_grad()
            x_samples = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((num_samples,))
            x_samples.requires_grad = True
            v = model(x_samples)
            Dv = get_gradient(v, x_samples)
            Lap_v = get_laplacian(Dv, x_samples)
            drift = torch.matmul(x_samples, self.A)
            forcing = torch.sum(x_samples * x_samples, axis = 1).reshape(num_samples, 1)
            b_Dv = torch.sum(Dv * drift, axis = 1).reshape(num_samples, 1)

            mu_samples = self.mu.sample((num_mu_samples,))
            aB = torch.matmul(mu_samples, self.B)
            stretch_Dv= Dv.repeat_interleave(num_mu_samples,0).reshape(num_samples,num_mu_samples,2)
            aNa = 0.5 * torch.sum(torch.matmul(mu_samples, self.N) * mu_samples, axis = 1)
            int_mu = torch.logsumexp(-(torch.sum(stretch_Dv * aB,axis = 2) + aNa) / self.tau , dim =1).reshape(num_samples, 1)

            pde = 0.5 * self.sigma**2 * Lap_v - self.rho * v + b_Dv + 0.5 * forcing - self.tau * int_mu + self.tau*np.log(num_mu_samples)
 
            target_functional = torch.zeros_like(v)
            MSE_functional = loss_fn(pde, target_functional)
            MSE_convexity = loss_fn(pde, torch.sum(x_samples * x_samples, axis = 1).reshape(num_samples,1))
            learning_loss.append(MSE_functional.item())

            exact_values = self.exact_sol_exp_HJB(x_samples)
            mre = torch.mean(mean_rel_error(exact_values,v))
            learning_loss_against_exact.append(mre)

            loss = MSE_functional + MSE_convexity/(epoch+1)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            if epoch%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}".format(epoch, num_of_epochs, MSE_functional.item()))
            if np.sum(learning_loss[-30:]) / 20 <= 0.002:
                break
        return(model, learning_loss, learning_loss_against_exact)
        
            
    def exact_sol_exp_HJB(self,x_points):
        C_tau = (2* (0.3)**2) / (8+5*self.tau)  # correct
        k = (0.6 - self.rho + np.sqrt( (self.rho - 0.6)**2 + 8*C_tau)) / (8*C_tau) # correct
        constant = (2 * (self.sigma**2) * k + self.tau * (np.log( (8+5*self.tau) / (5*self.tau))) ) / self.rho
        output = k * torch.sum(x_points * x_points, axis = 1) + constant
        return(output.reshape((len(output),1)))
    
    def MC_value_function_approx(self, x, MC, steps, dt, value_funct_model, num_mu_samples):
        sigma = self.tau * torch.inverse(self.N + self.tau * torch.eye(2))
        sigma_batch = sigma.expand(MC,2,2)
        def control_sampler(x,cov,grad):
            '''
            this function takes a vector {x_i}_{i=1}^N and return {a_i}_{i=1}^N where a_i sampled from \pi^*(da|x_i)
            '''
            mu = -torch.matmul(grad,torch.matmul(self.B, sigma))
            return(multi_norm_sampler.MultivariateNormal(mu, cov).sample())
        integral = 0
        X0 = x.repeat(MC,1)
        X0.requires_grad = True
        pbar = tqdm(total = steps)
        for i in range(steps):
            value_fnct_values = value_funct_model(X0)
            grads = get_gradient(value_fnct_values, X0)
            with torch.no_grad():
                A_samples_from_X0 = control_sampler(x,sigma_batch,grads)

                xMx_2 = 0.5 * torch.sum(torch.matmul(X0, self.M) * X0, axis =1)
                aNa_2 = 0.5 * torch.sum(torch.matmul(A_samples_from_X0, self.N) * A_samples_from_X0, axis = 1)

                mu_samples = self.mu.sample((num_mu_samples,))
                aB = torch.matmul(mu_samples, self.B)
                stretch_Dv= grads.repeat_interleave(num_mu_samples,0).reshape(MC,num_mu_samples,2)
                aNa = 0.5 * torch.sum(torch.matmul(mu_samples, self.N) * mu_samples, axis = 1)
                int_mu = torch.logsumexp(- (torch.sum(stretch_Dv * aB, axis = 2)+aNa) /self.tau, dim =1)

                discount = np.exp(-self.rho * i * dt)
                integral = integral + discount * (dt/MC) * torch.sum(xMx_2 + aNa_2 - self.tau * int_mu + self.tau * np.log(num_mu_samples), axis = 0).item()

            b_X0 = torch.matmul(X0, self.A) + torch.matmul(A_samples_from_X0, self.B) 
            X1 = X0 + dt * b_X0 + self.sigma * np.sqrt(dt) * torch.randn(MC,2)
            X0 = X1
            pbar.update(1)
        return(integral)

    

    '''    
    def generate_from_state_space(self, N):
        
        generate from the uniform distribution over the state space
        which we take to be the ball of radius rad
        
        rand_rad =  5 * torch.sqrt(torch.rand(N))
        rand_ang =  2 * np.pi * torch.rand(N)
        x_cord = rand_rad * np.cos(rand_ang)
        y_cord = rand_rad * np.sin(rand_ang)
        return(torch.stack((x_cord,y_cord), 1))
    
    def generate_from_boundary_state_space(self, N):
        
        #generate from the uniform distribution over the state space
        #which we take to be the ball of radius rad
        
        rand_ang =  2 * np.pi * torch.rand(N)
        x_cord = 5 * np.cos(rand_ang)
        y_cord = 5 * np.sin(rand_ang)
        return(torch.stack((x_cord,y_cord), 1))
    def learn_on_policy_bellman_for_policy_mu(self, dimX, dimH, lr, num_epochs,  num_samples):
        
        num_samples
        
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
            
            loss = MSE_functional 
            
            loss.backward()
            optimizer.step()
            pbar.update(1)

            if (epoch+1) % 20 == 0:
                print(epoch, np.mean(learning_loss[-10:]))



        return(model,learning_loss)
    '''


    


    




