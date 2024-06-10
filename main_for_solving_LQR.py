import torch
import torch.nn as nn
import LQR_elliptic
import NN_for_PDEs
import matplotlib.pyplot as plt
import torch.distributions.multivariate_normal as multi_norm_sampler
import numpy as np
import itertools


if __name__ == '__main__':
    A = torch.eye(2).float() * 0.3
    B = torch.eye(2).float() * 0.3
    sigma = 1 * 0.8
    M = torch.eye(2).float() 
    N = torch.eye(2).float() * 2 # N needs to be multiple of identity
    rho = 1.4 # discounting
    tau = 0.5 # regulariser 
    mu = multi_norm_sampler.MultivariateNormal(torch.zeros(2), 0.8 * torch.eye(2))
    LQR = LQR_elliptic.LQR_problem_reg(A, B, sigma, M, N, rho, tau, mu)
    
    learning_Rate = 0.01
    num_samples_mu = 1200

    # learning the uncontrolled setting
    '''
    uncontrolled_PDE_solver = LQR.learn_uncontrolled_problem(dimX = 2, dimH = 100, lr = learning_Rate, num_of_epochs = 4500,num_samples = 500)
    torch.save(uncontrolled_PDE_solver[0].state_dict(), 'trained_models/trained_cost_functional_no_control.pth')
    plt.plot([i for i in range(len(uncontrolled_PDE_solver[1])-30)], np.log(uncontrolled_PDE_solver[1][30:]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for cost functional with control no control')
    plt.savefig("training loss_functional_no_control.pdf")
    plt.close()
    '''
    # uncontrolled MC approximation
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
    '''
    # Now learn the cost functional for the control = mu
    '''
    '''
    model_with_control_mu = LQR.learn_cost_funct_control_mu(dimX=2,dimH=100,lr=learning_Rate,num_of_epochs=3000,num_samples=500,num_mu_samples=num_samples_mu,mu=mu)
    print(model_with_control_mu[0](torch.tensor([[ 0.0897, -0.7171]])))
    torch.save(model_with_control_mu[0].state_dict(), 'trained_models/trained_cost_functional_with_control_mu.pth')
    plt.plot([i for i in range(len(model_with_control_mu[1]))], np.log(model_with_control_mu[1]), linewidth = 1)    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for cost functional with control mu')
    plt.savefig("training_loss_for_cost_funct_with_control_mu.pdf")
    plt.close()
    '''
    # control = mu MC approximation
    '''
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
    '''
    
    # Now learn the value function
    model_value_function = LQR.learn_value_function(dimX = 2,dimH = 100,lr=learning_Rate,num_of_epochs=2500,num_samples=500,num_mu_samples=num_samples_mu)
    torch.save(model_value_function[0].state_dict(), 'trained_models/trained_optimal_value_funct.pth')
    
    # plot training loss for PDE againt 0
    plt.plot([i for i in range(len(model_value_function[1]))],np.log(model_value_function[1]))    
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE.')
    plt.title('training loss for value function')
    plt.savefig("training_loss_for_value_function.pdf")

    # plot training loss against exact solution from hand
    plt.plot([i for i in range(len(model_value_function[2]))],np.log(model_value_function[1]))    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Relative Error against exact solution.')
    plt.title('training loss for value function against the exact solution')
    plt.savefig("training_loss_for_value_function_against_exact_sol.pdf")


    # Graphing the value function too check it looks like x^2
    model_value_funct = NN_for_PDEs.Net_DGM(dim_x=2, dim_S=100)
    model_value_funct.load_state_dict(torch.load('trained_models/trained_optimal_value_funct.pth')) 
    model_value_funct.eval()
    model_value_funct.requires_grad = True
    num_points = 200
    spatial_grid_1D = torch.linspace(-6,6,num_points)
    x,y = np.meshgrid(spatial_grid_1D,spatial_grid_1D) 
    x_flat = x.flatten()
    y_flat = y.flatten()
    xy = np.vstack((x_flat, y_flat)).T
    xy_tensor = torch.tensor(xy, dtype=torch.float32)
    
    with torch.no_grad():
        z_vals = model_value_funct(xy_tensor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,z_vals.reshape(num_points,num_points), cmap='viridis')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_zlabel('v(x,y)')
    ax.set_title('Plot of learnt solution to HJB')
    plt.savefig('value_function_surface_plot.png', dpi=300)  # Change the filename and dpi as needed
    plt.show()
    
    #Testing the value function against the exact solution
    exact_solutions = LQR.exact_sol_exp_HJB(xy_tensor)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,exact_solutions.reshape(num_points,num_points), cmap='viridis')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_zlabel('v(x,y)')
    ax.set_title('Plot of exact solution to HJB')
    plt.savefig('value_function_exact_sol_surface_plot.png', dpi=300)  # Change the filename and dpi as needed
    plt.show()

    def mean_rel_error(tensor_1,tensor_2):
        mre = abs(tensor_1-tensor_2) / abs(tensor_1)
        return(mre)
    print('MSE between the exact opitmal value function and the trained model: {} '.format(torch.mean(mean_rel_error(exact_solutions, z_vals))))
    # plot the mean relatice errors
    mre = mean_rel_error(exact_solutions, z_vals)
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, mre.reshape(num_points,num_points), cmap='viridis', levels=50)
    plt.colorbar(label='realative error')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.savefig('value_function_mean_rel_error.png', dpi=300)
    plt.show()



    
    

    #monte_carlo_error = []
    #max_steps = 45000
    #step_size = 0.0002
    #x = torch.tensor([[ 0.0897, -0.7171]])
    #NN_value = model_value_funct(x)
    #print("x = {}:\t value from model = {:.4f}".format(x, NN_value.item()))
    #MC_list = [5000] #[10, 50, 100, 500, 1000, 5000, 10000, 50000]
    #num_samples_mu = 400
    #for MC in MC_list:
    #    MC_approx_control_mu = LQR.MC_value_function_approx(x=x,MC=MC,steps=max_steps,dt=step_size,
    #                                                        value_funct_model=model_value_funct,
    #                                                        num_mu_samples=num_samples_mu)
    #    relative_error = abs(NN_value.item() - MC_approx_control_mu) / abs(NN_value.item())
    #    monte_carlo_error.append(abs(NN_value.item() - MC_approx_control_mu) / abs(NN_value.item()))
    #    print('For MC = {}\t the MC approximation is {} \tRelative Error = {}'.format(MC,MC_approx_control_mu,relative_error))
    #plt.plot(np.log(MC_list), np.log(monte_carlo_error), marker = 'o')    
    #plt.xlabel('Log Number of MC samples')
    #plt.ylabel('Log Relative Error.')
    #plt.title('Monte Carlo Error, with {} steps of size {}'.format(max_steps,step_size))
    #plt.savefig("MC_Error_constant_mu.pdf")
    #plt.close()







    



    
