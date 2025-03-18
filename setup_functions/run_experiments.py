"""
creates data and hyperparameters to pass to neural network trainer
trains multiple times, changing one hyperparameter to compare impact of that hyperparameter
"""

import torch
import numpy as np # ali: for seed test
from setup_functions.pre_experiment import parse_voc_file
from setup_functions.vensim_simulation_model import vensim_simulation_model
from setup_functions.run_analysis import run_analysis
from sbi import utils as utils
import pandas as pd
import os

# ali: added seed
np.random.seed(42)
torch.manual_seed(42)

# function for running experiments and collecting data and also saving it
def run_experiments(base_vensimInputs, base_SBI_Inputs, base_resInputs, experiments, save_path, update_manual_data=False):
    """
    runs experiments with different conditions and collects outputs in a DataFrame, including full ndarrays.
    
    Parameters
    ----------
    base_vensimInputs : dict
        dictionary with baseline vensim input values
    base_SBI_Inputs : dict
        dictionary with baseline SBI input values
    base_resInputs : dict
        dictionary with baseline res input values
    experiments : list
        list of dictionaries specifying the experimental conditions
    save_path : string
        path to save the results DataFrame
    use_manual_data : Boolean
        if experiment updates vensimInputs['manual_summaries']
    
    Returns:
    - DataFrame with the results of all experiments.
    """
    if not update_manual_data: #manual data is not updated, so we can use one set of experiment data
        base_vensimInputs['venParameters'] = parse_voc_file(base_vensimInputs)
        lows = [val[0] for val in base_vensimInputs['venParameters'].values()]
        highs = [val[1] for val in base_vensimInputs['venParameters'].values()]
        prior = utils.BoxUniform(low=torch.tensor(lows, dtype=torch.float16, device=base_vensimInputs['device']),
                                 high=torch.tensor(highs, dtype=torch.float16, device=base_vensimInputs['device']))
    
        theta=prior.sample((base_SBI_Inputs['sampleSize'],))
        updated_theta, experiment_data = vensim_simulation_model(theta, base_vensimInputs, lows, highs)
    else: #initialize and make none; can pass as parameters, will catch errors
        experiment_data, lows, highs, prior, updated_theta = (None, None, None, None, None)
    all_results = []  # Initialize a list to store experiment results
    
    for exp in experiments:
        # Copy baseline inputs
        vensimInputs = base_vensimInputs.copy()
        SBI_Inputs = base_SBI_Inputs.copy()
        resInputs = base_resInputs.copy()
        
        # Update inputs based on experiment conditions
        vensimInputs.update(exp.get('vensim_changes', {}))
        SBI_Inputs.update(exp.get('SBI_changes', {}))
        resInputs.update(exp.get('res_changes', {}))
        
        # Run the simulation
        outputs = run_analysis(vensimInputs, SBI_Inputs, resInputs, data=experiment_data,
                                    lows=lows, highs=highs, prior=prior, theta=updated_theta, update_manual_data=update_manual_data)
        
        # Prepare the result dictionary
        result = {'experiment_label': exp['label']}
        result.update(outputs)
        
        # Arrays will be stored directly in the DataFrame
        all_results.append(result)
    
    print('Done with experiments')
    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(all_results)
    directory = os.path.dirname(save_path)  # Extracts the directory part of the save_path
    
    # Check if the directory exists; if not, create it
    if resInputs['save_pickle']:
        if not os.path.exists(directory):
            os.makedirs(directory)  # This creates the directory and any intermediate directories
            
        # Save the DataFrame to disk for future retrieval
        results_df.to_pickle(save_path)
    
    return results_df