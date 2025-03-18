#run_analysis.py
"""
full pipeline for training a neural network
- generates manual summary statistics if necessary
- creates embedding networks
- traines neural network, and accumulates metadata
- plots diagnostic graphs, if desired

also includes a function to get the true parameters, which are used in plotting
"""


import torch
import numpy as np
from setup_functions.pre_experiment import parse_voc_file
from setup_functions.vensim_simulation_model import vensim_simulation_model, refit_prior_bounds
from setup_functions.classes import SequenceNetwork
from setup_functions.plots import plot_posterior, plot_ground_truth, plot_z_score, plot_confidence_intervals
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
import matplotlib.pyplot as plt
import io
import sys
import time
import sys

np.random.seed(42)
torch.manual_seed(42)


def generate_true_parameters(prior, posterior, vensimInputs, SBI_Inputs, lows, highs):
    """
    generates true parameters from given prior distribution, simulates observations using vensim_simulation_model; then infers parameters using posterior

    Parameters
    ----------
    prior : torch.distributions.Distribution
        prior distribution to sample the true parameters from
    posterior : sbi.inference.posteriors.DirectPosterior
        posterior distribution to infer parameters from the observations
    vensimInputs : dict
        dictionary containing inputs required for the Vensim simulation model; keys include:
        - 'device': str
            device to use for tensor computations (e.g., 'cpu' or 'cuda').

    Returns (###this should be updated###)
    -------
    tuple of numpy.ndarray
        tuple containing two elements:
        - posterior_samples: numpy.ndarray
            array of posterior samples, shape (n_samples, 1000, param_dim);`n_samples` is number of simulations, `param_dim` is dimensionality of parameters.
        - true_params: numpy.ndarray
            array of true parameters sampled from the prior distribution with shape (n_samples, param_dim)
    """

    # oops. we've done a mistake here. true parameters should be bounded.
    #true_params_tensor = prior.sample((SBI_Inputs['valid_size'],))
    #true_params_tensor = refit_prior_bounds(true_params_tensor, lows, highs)
    
    # and we actually should use a box uniform prior for true_params. otherwise, the clipped true params would not be a good representative
    true_params_tensor = torch.distributions.Uniform(low=torch.tensor(lows), high=torch.tensor(highs)).sample((SBI_Inputs['valid_size'],))

    # if we want to calculate the metrics for GB for a single validatoin dataset:
    #true_params_tensor = torch.tensor([0.130952, 0.00654762, 0.0514706, 13.4259, 0.0120536, 0.0425, 0.0125, 0.166667, 3.85]).unsqueeze(0).to(vensimInputs['device'])

    # Simulate observations using these fixed true parameters
    observations = []
    for params in true_params_tensor:
        _, x_o = vensim_simulation_model(params.unsqueeze(0), vensimInputs, lows, highs)
        x_o = x_o.to(vensimInputs['device'])
        observations.append(x_o)
    observations = torch.stack(observations)

    # For each observation, infer posterior samples
    posterior_samples = []
    for x_o in observations:
        samples = posterior.sample((SBI_Inputs['post_samples_per_obs'],), x=x_o, show_progress_bars=False)
        posterior_samples.append(samples.cpu().numpy())
    posterior_samples = np.array(posterior_samples)
    true_params = true_params_tensor.cpu().numpy()

    return posterior_samples, true_params, observations, true_params_tensor


def run_analysis(vensimInputs, SBI_Inputs, resInputs, data, lows, highs, prior, theta, update_manual_data):
    """
    runs full analysis pipeline, including prior sampling, simulation, inference, and visualization

    Parameters
    ----------
    vensimInputs : dict
        dictionary with inputs and data relevant for vensim_simulation_function
    SBI_Inputs : dict
        dictionary with inputs relevant for embedding network classes and SNPE trainer
    resInputs : dict
        dictionary with inputs relevant for visualization
    data : torch.Tensor
        data to be used for inference
    lows : list of float
        lower bounds for parameters
    highs : list of float
        upper bounds for parameters
    prior : torch.distributions.Distribution
        prior distribution for the parameters
    theta : torch.Tensor
        sampled parameters from the prior distribution
    update_manual_data : bool
        flag indicating whether to update manual data; if true, Vensim parameters will be parsed and used for sampling

    Returns
    -------
    runOutputs : dict
        dictionary containing runtime outputs including training time and loss value
    """
    if update_manual_data: #true if experiment updates vensimInputs['manual_data']; need unique simulations
        vensimInputs['venParameters'] = parse_voc_file(vensimInputs)
        lows = [val[0] for val in vensimInputs['venParameters'].values()]
        highs = [val[1] for val in vensimInputs['venParameters'].values()]
        prior = utils.BoxUniform(low=torch.tensor(lows, dtype=torch.float32, device=vensimInputs['device']),
                                 high=torch.tensor(highs, dtype=torch.float32, device=vensimInputs['device']))
        theta = prior.sample((SBI_Inputs['sampleSize'],))
        updated_theta, data = vensim_simulation_model(theta, vensimInputs, lows, highs)
        
    SBI_Inputs['initial_in_channels'] = data.size(-1)
    sequence_net = SequenceNetwork(vensimInputs,
                                   summary_dim=SBI_Inputs['summary_dim'],
                                   num_conv_layers=SBI_Inputs['num_conv_layers'],
                                   lstm_units=SBI_Inputs['lstm_units'],
                                   bidirectional=SBI_Inputs['bidirectional'],
                                   out_channels=SBI_Inputs['out_channels'],
                                   min_kernel_size=SBI_Inputs['min_kernel_size'],
                                   max_kernel_size=SBI_Inputs['max_kernel_size'],
                                   initial_in_channels=SBI_Inputs['initial_in_channels']
                                  )
    if SBI_Inputs['model'] == 'maf':
        inference = SNPE(prior=prior, device=vensimInputs['device'], density_estimator=posterior_nn(
                             model='maf',
                             embedding_net=sequence_net,
                             hidden_features=SBI_Inputs['hidden_features'],
                             num_transforms=SBI_Inputs['num_transforms'],
                             num_blocks=SBI_Inputs['num_blocks'],
                             dropout_probability=SBI_Inputs['dropout_probability'],
                             use_batch_norm=SBI_Inputs['use_batch_norm'])
                        )
    else:
        inference = SNPE(prior=prior, device=vensimInputs['device'],
                         density_estimator=posterior_nn(model=SBI_Inputs['model'], embedding_net=sequence_net)
                        )

    inference=inference.append_simulations(updated_theta, data)
    runOutputs={}

    #this is the only way I know how to extract the "Best Validation Performance"
    train_output = io.StringIO()
    sys.stdout = train_output
    start_time=time.time()
    density_estimator=inference.train(training_batch_size=SBI_Inputs['training_batch_size'],
                                      stop_after_epochs=SBI_Inputs['stop_after_epochs'],
                                      learning_rate=SBI_Inputs['learning_rate'],
                                      show_train_summary=SBI_Inputs['show_train_summary']
                                     )
    end_time=time.time()
    train_output_string = train_output.getvalue()
    start_index = train_output_string.find('e:')
    validation_performance = train_output_string[start_index+2:start_index+10].strip()
    posterior = inference.build_posterior(density_estimator)
    
    runOutputs['train_time'] = end_time-start_time
    runOutputs['loss_value'] = float(validation_performance)


    if resInputs['graph_all'] and resInputs['graph_set_up']:
        posterior_samples, true_params = generate_true_parameters(prior=prior, posterior=posterior, vensimInputs=vensimInputs, valid_size=SBI_Inputs['valid_size'], post_samples_per_obs=SBI_Inputs['post_samples_per_obs'], lows=lows, highs=highs) #it took me a while to understand that i should update this 
    
    if resInputs['graph_all'] and resInputs['graph_posterior']:
        plot_posterior(posterior=posterior,
                       lows=lows, highs=highs,
                       vensimInputs=vensimInputs,
                       save_graph=resInputs['save_graphs'],
                       show_graph=resInputs['show_graphs'],
                       save_name=resInputs['scenario_name']
                      )
    if resInputs['graph_all'] and resInputs['graph_recovery']:
        plot_ground_truth(post_samples=posterior_samples,
                          true_params=true_params,
                          param_names=list(vensimInputs['venParameters'].keys()),
                          save_graph=resInputs['save_graphs'],
                          show_graph=resInputs['show_graphs'],
                          save_name=resInputs['scenario_name']
                         )
    if resInputs['graph_all'] and resInputs['graph_z_score']:
        plot_z_score(post_samples=posterior_samples,
                     true_params=true_params,
                     param_names=list(vensimInputs['venParameters'].keys()),
                     save_graph=resInputs['save_graphs'],
                     show_graph=resInputs['show_graphs'],
                     save_name=resInputs['scenario_name']
                    )
    if resInputs['graph_all'] and resInputs['graph_confidence']:
        plot_confidence_intervals(post_samples=posterior_samples,
                                  true_params=true_params,
                                  param_names=list(vensimInputs['venParameters'].keys()),
                                  save_graph=resInputs['save_graphs'],
                                  show_graph=resInputs['show_graphs'],
                                  save_name=resInputs['scenario_name']
                                 )
    return runOutputs