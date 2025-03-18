import torch
import numpy as np
from setup_functions.pre_experiment import write_strings_to_file, create_tab_delimited, extract_sensitivities, generate_summary_statistics
import venpy
import os


np.random.seed(42)
torch.manual_seed(42)

def refit_prior_bounds(theta, lows, highs, epsilon=0.0001):
    """
    Truncates the theta samples to the prior boundaries. If the lows are 0, then add
    epsilon to ensure simulation occurs, does not simulate all values to 0, as if
    simulation never happens

    -----------
    Parameters:
        - theta (tensor): random sample
        - lows (list): low value for each parameter of model
        - highs (list): high value for each parameter of model
        - epsilon (list): value above 0 to set low values (default: 0.0001)
    
    Returns:
        - (tensor) truncated samples
    """

    theta = theta.unsqueeze(0) if theta.dim() == 1 else theta
    lows = torch.where(lows==0, lows+epsilon, lows) # if lows is 0, add epsilon to move away from boundary
    highs = torch.where(highs==0, highs-epsilon, highs)
    #print("lows: ", lows)
    #print("highs: ", highs)

    theta = torch.clamp(theta, min=lows, max=highs)
    
    return theta

def vensim_simulation_model(theta, vensimInputs, lows, highs):
    """
    Performs a batch of forward simulations from the SD model given a batch of
    random draws from the prior
    
    -----------
    Parameters:
        - theta (tensor): random sample
        - vensimInputs (dictionary): customizable settings pertaining to simulating
        - lows (list): low value for each parameter of model
        - highs (list): high value for each parameter of model

    Returns:
        - (tensor) truncated samples
        - (tensor) simulated data
    """

    updated_theta = refit_prior_bounds(theta, lows, highs)
    updated_theta = updated_theta.cpu().numpy()
    #print("updated_theta: ", updated_theta)
    # trying to save the parameters
    # it saves all parameters without their names in a tab-delimited fomrat
    # which can be used for sensitivity analysis inside vensim
    # note: vensim doesn't throw same errors as dll, which is a bug
    # so this might not be very helpful if vensim doesn't raise the same error
    # i've already reported this to vensim devs, tom, tony etc. 
    # np.savetxt("updated_theta.txt", updated_theta, delimiter='\t')
    #print("updated_theta: ", updated_theta)
    venModelPub = vensimInputs['venModelName'] + '.vpmx'
    venSensFile = "SIRSens.txt"
    venSaveList = 'sens.lst'
    venSensControl = "Sens.vsc"
    seed_start = np.random.randint(0, 10000000)
    ven_param_names = list(vensimInputs['venParameters'].keys())
    ven_const_names = vensimInputs['constants']
    Constants = vensimInputs['constant_vals']
    output_vars = vensimInputs['outputs']
    Times = vensimInputs['Time_points']
    venSeedVar = vensimInputs['SeedVar']

    #it generates all updated_theta that goes into vensim dll 
    #in a clean format with simulation numbers that can be used 
    #as .cin for vensim inputs
    #let's keep a version of this for diagnostic purposes
    
    # updated_theta = refit_prior_bounds(theta, lows, highs)
    # updated_theta = updated_theta.cpu().numpy()
    # # The file will be written later after obtaining parameter names
    # #print("updated_theta: ", updated_theta)
    # venModelPub = vensimInputs['venModelName'] + '.vpmx'
    # venSensFile = "SIRSens.txt"
    # venSaveList = 'sens.lst'
    # venSensControl = "Sens.vsc"
    # seed_start = np.random.randint(0, 10000000)
    # ven_param_names = list(vensimInputs['venParameters'].keys())
    # # Save updated_theta to a text file with parameter names for each simulation
    # with open("updated_theta.txt", "w") as f:
    #     for i, row in enumerate(updated_theta):
    #         for param, val in zip(ven_param_names, row):
    #             f.write("S{} {} = {}\n".format(i+1, param, val))
    #         f.write("\n")
    # ven_const_names = vensimInputs['constants']
    # Constants = vensimInputs['constant_vals']
    # output_vars = vensimInputs['outputs']
    # Times = vensimInputs['Time_points']
    # venSeedVar = vensimInputs['SeedVar']
    
    write_strings_to_file(output_vars, venSaveList)
    write_strings_to_file(['200,F,1234,' + os.getcwd() + '/' + venSensFile + ',0'], venSensControl)
    create_tab_delimited(updated_theta, ven_param_names, venSensFile, seed_start, SeedVar=venSeedVar)

    # Read the model
    venmodel = venpy.load(venModelPub)
    for i, varNm in enumerate(ven_const_names):
        venmodel[varNm] = Constants[i]
        
    # Run simulations
    venmodel.run(runname='sens', sensitivity=[venSensControl, venSaveList])
    sim_data = extract_sensitivities(venmodel, output_vars, Times, updated_theta.shape[0], SeedVar=venSeedVar)
    updated_theta = torch.from_numpy(updated_theta).to(vensimInputs['device'])
    num_sims, num_time, num_series = sim_data.shape

    if vensimInputs['manual_summaries']:
        # Initialize array for new sim_data with manual summaries
        for i in range(num_sims):
            sim_data_i = sim_data[i]  # Shape: (num_time, num_series)
            sim_data_transposed = sim_data_i.T  # Shape: (num_series, num_time)
            summaries= generate_summary_statistics(sim_data_transposed, vensimInputs)  # Shape: (num_series, num_summaries_per_series)
            # Concatenate summaries to each corresponding time series
            sim_data_with_summaries = np.concatenate((sim_data_transposed, summaries), axis=1)  # Shape: (num_series, num_time + num_summaries_per_series)
            sim_data_with_summaries_t = sim_data_with_summaries.T  # Shape: (num_time + num_summaries_per_series, num_series)

            if i == 0:
                new_time_dim = sim_data_with_summaries_t.shape[0]
                sim_data_with_manual = np.empty((num_sims, new_time_dim, num_series))
            sim_data_with_manual[i] = sim_data_with_summaries_t

        # Convert to torch tensor
        sim_data_with_manual = torch.from_numpy(sim_data_with_manual).float()
        torch.set_printoptions(threshold=float('inf'))
        return updated_theta, sim_data_with_manual
    else:
        # Return sim_data as torch tensor
        sim_data = torch.from_numpy(sim_data).float()
        return updated_theta, sim_data

