import torch
import LQR_elliptic
import NN_for_PDEs
import matplotlib.pyplot as plt
import torch.distributions.multivariate_normal as multi_norm_sampler
import numpy as np


if __name__ == '__main__':
    A = torch.eye(2).float() * 0.3
    B = torch.eye(2).float() * 0.3
    sigma = 1 * 0.8
    M = torch.tensor([[1, 0], [0,1]])
    N = torch.eye(2).float() * 2 # N needs to be multiple of identity
    rho = 1.4 # discounting
    tau = 0.5 # regulariser 
    LQR = LQR_elliptic.LQR_problem_reg(A, B, sigma, M, N, rho, tau)
    
    learning_Rate = 0.01
    num_samples_mu = 1200

    '''
    uncontrolled_PDE_solver = LQR.learn_uncontrolled_problem(dimX = 2, dimH = 100, lr = learning_Rate, num_of_epochs = 4500,num_samples = 500)
    torch.save(uncontrolled_PDE_solver[0].state_dict(), 'trained_models/trained_cost_functional_no_control.pth')
    plt.plot([i for i in range(len(uncontrolled_PDE_solver[1])-30)], np.log(uncontrolled_PDE_solver[1][30:]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for cost functional with control no control')
    plt.savefig("trained_cost_functional_no_control.pdf")
    plt.close()
    '''
    # Now do the MC approximation
    '''
    model_no_control = NN_for_PDEs.Net_DGM(dim_x=2, dim_S=100)
    model_no_control.load_state_dict(torch.load('trained_models/trained_cost_functional_no_control.pth')) 
    model_no_control.eval()
    model_no_control.requires_grad = True

    monte_carlo_error = []
    max_steps = 60000
    step_size = 0.0002
    x = torch.tensor([[ 0.0897, -0.7171]])
    NN_value = model_no_control(x)
    print("x = {}:\t value from model = {:.4f}".format(x, NN_value.item()))
    MC_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    for MC in MC_list:
        MC_approx_no_control = LQR.MC_approx_no_control(x=x, MC=MC, steps=max_steps, dt=step_size)
        relative_error = abs(NN_value.item() - MC_approx_no_control) / abs(NN_value.item())
        monte_carlo_error.append(abs(NN_value.item() - MC_approx_no_control) / abs(NN_value.item()))
        print('For MC = {}\t the MC approximation is {} \tRelative Error = {}'.format(MC,MC_approx_no_control,relative_error))
    plt.plot(MC_list, np.log(monte_carlo_error), marker = 'o')    
    plt.xlabel('Number of MC samples')
    plt.ylabel('Log Relative Error.')
    plt.title('Monte Carlo Error, with {} steps of size {} number of steps'.format(max_steps,step_size))
    plt.savefig("MC_Error_constant_control.pdf")
    plt.close()
    '''
    
    model_with_control_mu = LQR.learn_cost_funct_control_mu(dimX=2,dimH=100,lr=learning_Rate,num_of_epochs=3000,num_samples=500,num_mu_samples=num_samples_mu)
    print(model_with_control_mu[0](torch.tensor([[ 0.0897, -0.7171]])))
    torch.save(model_with_control_mu[0].state_dict(), 'trained_models/trained_cost_functional_with_control_mu.pth')
    plt.plot([i for i in range(len(model_with_control_mu[1]))], np.log(model_with_control_mu[1]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for cost functional with control mu')
    plt.savefig("training_loss_for_cost_funct_with_control_mu.pdf")
    plt.close()
    model_value_funct_control_mu = NN_for_PDEs.Net_DGM(dim_x=2, dim_S=100)
    model_value_funct_control_mu.load_state_dict(torch.load('trained_models/trained_cost_functional_with_control_mu.pth')) 
    model_value_funct_control_mu.eval()
    model_value_funct_control_mu.requires_grad = True
    x = torch.tensor([[ 0.0897, -0.7171]])
    NN_value = model_value_funct_control_mu(x)
    monte_carlo_error = []
    max_steps = 45000
    step_size = 0.0002
    x = torch.tensor([[ 0.0897, -0.7171]])
    NN_value = model_value_funct_control_mu(x)
    print("x = {}:\t value from model = {:.4f}".format(x, NN_value.item()))
    MC_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    for MC in MC_list:
        MC_approx_control_mu = LQR.MC_approx_control_mu(x=x, MC=MC, steps=max_steps, dt=step_size)
        relative_error = abs(NN_value.item() - MC_approx_control_mu) / abs(NN_value.item())
        monte_carlo_error.append(abs(NN_value.item() - MC_approx_control_mu) / abs(NN_value.item()))
        print('For MC = {}\t the MC approximation is {} \tRelative Error = {}'.format(MC,MC_approx_control_mu,relative_error))
    plt.plot(np.log(MC_list), np.log(monte_carlo_error), marker = 'o')    
    plt.xlabel('Log Number of MC samples')
    plt.ylabel('Log Relative Error.')
    plt.title('Monte Carlo Error, with {} steps of size {} number of steps'.format(max_steps,step_size))
    plt.savefig("MC_Error_constant_mu.pdf")
    plt.close()
    

    #model_value_function = LQR.learn_value_function(dimX = 2, dimH = 100, lr = learning_Rate, 
    #                                                num_of_epochs = 2500, 
    #                                                num_samples = 500, 
    #                                                mu_samples = num_samples_mu)
    #torch.save(model_value_function[0].state_dict(), 'trained_optimal_value_funct.pth')
    #plt.plot([i for i in range(len(model_value_function[1])-30)], np.log(model_value_function[1][30:]), linewidth = 1)    
    #plt.xlabel('Epochs')
    #plt.ylabel('Log MSE.')
    #plt.title('training loss for value function')
    #plt.savefig("training_loss_for_value_function.pdf")
    #plt.close()

        #x = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2)).sample((1,))
    #print('*****')
    #x = torch.tensor([[ 0.0897, -0.7171]])
    #NN_value = model_value_funct_control_mu(x)
    #MC = 10000 # number of MC samples
    #T = 5
    #N_time = 5000 # number of time steps
    #dt = T / N_time 
    #mu_samples = 3000 # number of samples to evaluate integration against prior
    #SDE_samples = 3000
    #MC_approx = LQR.MC_control_mu_2(x,MC=MC, T=T, N_time=N_time, dt=dt, num_mu_samples=mu_samples, SDE_samples=SDE_samples)
    #MC_approx = LQR.MC_control_mu_2(x,MC=MC, T=T, N_time=N_time, dt=dt)
    #print(np.abs(NN_value.item() - MC_approx.item()) / abs(NN_value.item()))
    #print('****')






    



    
