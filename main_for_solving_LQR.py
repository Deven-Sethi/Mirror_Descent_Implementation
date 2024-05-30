import torch
import LQR_elliptic
import NN_for_PDEs
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    A = torch.tensor([[1, 0], [0, 1]])
    B = torch.eye(2).float() 
    sigma = 1
    M = torch.tensor([[1, 0], [0,1]])
    N = torch.eye(2).float() * 0.01 # N needs to be multiple of identity
    rho = 1.5 # discounting
    tau = 0.5 # regulariser 
    LQR = LQR_elliptic.LQR_problem_reg(A, B, sigma, M, N, rho, tau)
    
    learning_Rate = 0.003
    num_samples_mu = 4000
    model_value_function = LQR.learn_value_function(dimX = 2, dimH = 100, lr = learning_Rate, 
                                                    num_of_epochs = 2500, 
                                                    num_samples = 500, 
                                                    mu_samples = num_samples_mu)
    torch.save(model_value_function[0].state_dict(), 'trained_optimal_value_funct.pth')
    plt.plot([i for i in range(len(model_value_function[1])-30)], np.log(model_value_function[1][30:]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for value function')
    plt.savefig("training_loss_for_value_function.pdf")
    plt.close()

    model_with_control_mu = LQR.learn_cost_for_control_mu(dimX=2, dimH=100, lr=learning_Rate, 
                                                          num_of_epochs=2500, 
                                                          num_samples=500, 
                                                          mu_samples=num_samples_mu)
    torch.save(model_with_control_mu[0].state_dict(), 'trained_cost_functional_woith_control_mu.pth')
    plt.plot([i for i in range(len(model_with_control_mu[1])-30)], np.log(model_with_control_mu[1][30:]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for cost functional with control mu')
    plt.savefig("training_loss_for_cost_funct_with_control_mu.pdf")
    plt.close()

    

    model_value_funct_control_mu = LQR.learn_on_policy_bellman_for_policy_mu(dimX=2, dimH=100, lr=learning_Rate, num_epochs=3000, num_samples=500)
    model_value_funct_control_mu = NN_for_PDEs.Net_DGM(dim_x=2, dim_S=100)
    model_value_funct_control_mu.load_state_dict(torch.load('trained_policy_const_mu.pth')) 
    model_value_funct_control_mu.eval()
    model_value_funct_control_mu.requires_grad = True

    #MC = 10000
    #N_time_steps = 6000
    #T = 5
    #x = torch.tensor([[1,2]]).float()
    #exact = value_funct_model(x).item()
    #MC_approx_value_funct = LQR.MC_value_function_approx(MC_samples=MC, total_time_Steps=N_time_steps, T=T, initial_cond=x, model=value_funct_model, exact_val=exact)

    #print(value_funct_model(x).item(), MC_approx_value_funct)





    



    
