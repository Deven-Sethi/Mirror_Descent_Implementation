import torch
import LQR_elliptic
import NN_for_PDEs
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    A = torch.tensor([[1, 2], [0, 1]])
    B = torch.tensor([[1, 0], [0, 1]]).float()
    sigma = 1
    M = torch.tensor([[1, 2], [2,1]])
    R = torch.tensor([[1, 0], [0,1]]) * 0
    N = torch.eye(2).float() * 2 # N needs to be multiple of identity
    P = torch.tensor([1, 2])
    Q = torch.tensor([1,2]) 
    rho = 1.3 # discounting
    tau = 0.5 # regulariser 
    LQR = LQR_elliptic.LQR_problem_reg(A, B, sigma, M, R, N, P, Q, rho, tau)
    
    
    learning_Rate = 0.01
    num_samples_mu = 5000
    #model_value_function = LQR.learn_value_function(dimX = 2, dimH = 100, lr = learning_Rate, 
    #                                                num_of_epochs = 2500, num_samples = 500, 
    #                                                mu_samples = num_samples_mu)
    
    #torch.save(model_value_function[0].state_dict(), 'trained_optimal_value_funct.pth')

    #plt.plot([i for i in range(2500)], np.log(model_value_function[1][30:]), linewidth = 1)    plt.xlabel('Epochs')
    #plt.ylabel('Log MSE.')
    #plt.title('training loss for value function')
    #plt.savefig("training_loss_for_value_function.pdf")
    #plt.close()

    value_funct_model = NN_for_PDEs.Net_DGM(dim_x = 2, dim_S = 100)
    value_funct_model.load_state_dict(torch.load('trained_optimal_value_funct.pth')) 
    value_funct_model.eval()
    value_funct_model.requires_grad = True


    
    MC = 10000
    N_time_steps = 6000
    T = 5
    x = torch.tensor([[-1,2]]).float()
    exact = value_funct_model(x).item()
    MC_approx_value_funct = LQR.MC_value_function_approx(MC_samples=MC, total_time_Steps=N_time_steps, T=T, initial_cond=x, model=value_funct_model, exact_val=exact)

    print(value_funct_model(x).item(), MC_approx_value_funct)





    



    
