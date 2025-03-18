"""
obsolete functions, not used for SBI code
"""
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

def model_prior(variables):
    """
    generates a random draw from the joint prior
    
    -----------
    Parameters:
        - variables (dictionary): model parameter names and ranges
        
    Returns:
        - (numpy.ndarray) random draws from the priors
    """

    random_values = np.array([
        np.random.uniform(low, high) for name, (low, high) in variables.items()
    ])
    return torch.from_numpy(random_values)


# generating priors samples in two different format for vensim and torch
def generate_priors(sampleSize, variables):
    """
    generate prior samples
    must also have function 'model_prior' in file

    -----------
    Parameters:
        - sampleSize (int): number of samples to generate
        - variables (dictionary): model parameter names and ranges

    Returns:
        (tensor): tensor containing the generated prior values, converted to float
        (numpy.ndarray): array containing the generated prior values
    """
    
    prior_vals=np.empty((sampleSize, len(variables)))
    for i in range(sampleSize):
        prior_vals[i, :] = model_prior(variables)
    
    priors_tensor = torch.from_numpy(prior_vals)
    theta = priors_tensor.float()
    return theta, prior_vals