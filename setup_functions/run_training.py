#run_training.py
import torch
import numpy as np
from setup_functions.classes import SequenceNetwork
from setup_functions.run_analysis import generate_true_parameters
from setup_functions.plots import *
from setup_functions.pre_experiment import parse_voc_file
from setup_functions.prior import *
from setup_functions.validation_calculations import *
from setup_functions.vensim_simulation_model import *
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
import time
from torch.distributions import MultivariateNormal
import pickle

np.random.seed(42)
torch.manual_seed(42)

def data_setup(prior, vensimInputs, SBI_Inputs):
    """
    Create the sequence network and the inference object

    -----------
    Parameters:
        - prior (Distribution): Gaussian distribution of parameter values
        - vensimInputs (dictionary): customizable settings pertaining to simulating
        - SBI_Inputs (dictionary): customizable settings pertaining to training
    
    Returns:
        - (SNPE) inference object
    """

    sequence_net = SequenceNetwork(
        vensimInputs,
        summary_dim=SBI_Inputs['summary_dim'],
        num_conv_layers=SBI_Inputs['num_conv_layers'],
        lstm_units=SBI_Inputs['lstm_units'],
        bidirectional=SBI_Inputs['bidirectional'],
        out_channels=SBI_Inputs['out_channels'],
        min_kernel_size=SBI_Inputs['min_kernel_size'],
        max_kernel_size=SBI_Inputs['max_kernel_size'],
        initial_in_channels=SBI_Inputs['initial_in_channels']
    )    
    sequence_net.train()
    
    if SBI_Inputs['model'] == 'maf':
        inference = SNPE(prior=prior, device=vensimInputs['device'], density_estimator=posterior_nn(
                             model='maf',
                             embedding_net=sequence_net, 
                             hidden_features=SBI_Inputs['hidden_features'],
                             num_transforms=SBI_Inputs['num_transforms'],
                             num_blocks=SBI_Inputs['num_blocks'],
                             dropout_probability=SBI_Inputs['dropout_probability'],
                             use_batch_norm=SBI_Inputs['use_batch_norm'],)
                        )
    else:
        inference = SNPE(prior=prior, device=vensimInputs['device'],
                         density_estimator=posterior_nn(model=SBI_Inputs['model'],
                                                        embedding_net=sequence_net)
                        )
        
    return inference

def _train(prior, vensimInputs, SBI_Inputs, resInputs, lows=None, highs=None):
    """
    Trains a neural network in rounds. Then  plots results

    -----------
    Parameters:
        - prior (Distribution): Gaussian distribution of parameter values
        - vensimInputs (dictionary): customizable settings pertaining to simulating
        - SBI_Inputs (dictionary): customizable settings pertaining to training
        - res_Inputs (dictionary): customizable settings pertaining to plotting
        - lows (list): low value for each parameter of model
        - highs (list): high value for each parameter of model
    
    Returns:
        (Posterior) learned posterior object
    """

    # sbc became weird so i'm reverting to the original code we had

    # if vensimInputs['true_parameter'] is not None:
    #     true_parameter = vensimInputs['true_parameter'].to(vensimInputs['device'])
    # else:
    #     # is this coming from the truncated Gaussian distribution?
        
    #     # just for the paper, let's use a manual generation

    #     # Check if the model name is 'GeneralizedBass' and use a predefined tensor if so
    #     if vensimInputs.get('venModelName') == 'GeneralizedBass':
    #         true_parameter = torch.tensor([0.130952, 0.00654762, 0.0514706, 13.4259, 0.0120536, 0.0425, 0.0125, 0.166667, 3.85]).to(vensimInputs['device'])
    #     else:
    #         true_parameter = model_prior(vensimInputs['venParameters']).to(vensimInputs['device'])
    #         #print("true_parameter: ", true_parameter)

        
        
    #     # true_parameter = model_prior(vensimInputs['venParameters']).to(vensimInputs['device'])
    #     # print("true_parameter: ", true_parameter)
    #     # vensimInputs['true_parameter'] = true_parameter # stores true_parameter; use same true_parameter each round

    # if 'x_o' in vensimInputs and vensimInputs['x_o'] is not None:
    #     x_o = vensimInputs['x_o'].to(vensimInputs['device'])
    # else:
    #     vensimInputs['x_o'] = vensim_simulation_model(true_parameter.unsqueeze(0), vensimInputs, lows, highs)[1].to(vensimInputs['device'])
    #     x_o = vensimInputs['x_o'].to(vensimInputs['device'])

    if vensimInputs['true_parameter'] is not None:
        true_parameter = vensimInputs['true_parameter'].to(vensimInputs['device'])
    else:
        true_parameter = model_prior(vensimInputs['venParameters']).to(vensimInputs['device'])
        vensimInputs['true_parameter'] = true_parameter # stores true_parameter; use same true_parameter each round

    if 'x_o' in vensimInputs and vensimInputs['x_o'] is not None:
        x_o = vensimInputs['x_o'].to(vensimInputs['device'])
    else:
        vensimInputs['x_o'] = vensim_simulation_model(true_parameter.unsqueeze(0), vensimInputs, lows, highs)[1].to(vensimInputs['device'])
        x_o = vensimInputs['x_o'].to(vensimInputs['device'])

    SBI_Inputs['summary_dim'] = true_parameter.size(-1) * 4
    scenario_name = resInputs['scenario_name'] 
    resInputs['scenario_name'] = resInputs['scenario_name'] + f"_{(lambda s: s[::-1].replace('000', 'k')[::-1])(str(SBI_Inputs['sampleSize']))}"
    plot_time_series(x_o, vensimInputs, resInputs) if SBI_Inputs['amortized'] == False else None

    #testing the time series plot
    #plot_time_series(x_o, vensimInputs, resInputs)

    SBI_Inputs['initial_in_channels'] = x_o.size(-1)
    inference = data_setup(prior, vensimInputs, SBI_Inputs)
    #inference = NPE(prior=prior, device=vensimInputs['device'], density_estimator=posterior_nn(model='maf'))
    proposal = prior
    
    # Build the file name for saving/loading based on your naming rule.
    #density_estimator_file = f"{resInputs['scenario_name']}_NN.pickle"
    import os
    #print("Current directory:", os.getcwd())
    if SBI_Inputs['load_density_estimator']:
        density_estimator_file = os.path.join(os.getcwd(), 'density_estimators', f"{resInputs['scenario_name']}_NN.pickle")
        with open(density_estimator_file, 'rb') as f:
            density_estimator = pickle.load(f)
        posterior = inference.build_posterior(density_estimator, sample_with=SBI_Inputs['sample_with'])
    else:
        for current_round in range(SBI_Inputs['num_rounds']):
            theta = proposal.sample((SBI_Inputs['sampleSize'],)).to(vensimInputs['device'])
            _, outputs = vensim_simulation_model(theta, vensimInputs, lows, highs)
            outputs = outputs.to(vensimInputs['device'])
            #inference.append_simulations(theta, outputs, proposal=proposal)
            inference.append_simulations(theta, outputs)
            start_time = time.time()
            density_estimator = inference.train(training_batch_size=SBI_Inputs['training_batch_size'],
                                                stop_after_epochs=SBI_Inputs['stop_after_epochs'],
                                                learning_rate=SBI_Inputs['learning_rate'],
                                                show_train_summary=SBI_Inputs['show_train_summary'],
                                                force_first_round_loss=SBI_Inputs['force_first_round_loss'],
                                                use_combined_loss=SBI_Inputs['use_combined_loss'],
                                                discard_prior_samples=SBI_Inputs['discard_prior_samples'],
                                                retrain_from_scratch=SBI_Inputs['retrain_from_scratch'],
                                                )
            train_time = round(time.time()-start_time)
            #print(f"Training time: {train_time//3600}:{(train_time%3600)//60:02}:{train_time%60:02}\n\n")
            posterior = inference.build_posterior(density_estimator, sample_with=SBI_Inputs['sample_with'])

            if SBI_Inputs['num_rounds'] > 1:     
                if SBI_Inputs['amortized']:
                    SBI_Inputs['force_first_round_loss'] = True
                    proposal = proposal
                else:
                    proposal = posterior.set_default_x(x_o)
                inference._theta_roundwise = []
                inference._x_roundwise = []
                inference._proposal_roundwise = []
                inference._prior_masks = []
                inference._data_round_index = []

                if current_round in resInputs['rounds_to_plot_post']:
                    resInputs['scenario_name'] = resInputs['scenario_name'] + f'_round{current_round+1}'
                    plot_posterior(
                        posterior=posterior,
                        lows=lows, highs=highs,
                        vensimInputs=vensimInputs,
                        resInputs=resInputs,
                        x_o=x_o,
                        true_parameter=true_parameter
                    )
    
    SBI_Inputs['force_first_round_loss'] = False
    if SBI_Inputs['save_density_estimator']:
        with open(f"{resInputs['scenario_name']}_NN.pickle", 'wb') as f:
            pickle.dump(density_estimator, f)

    posterior_samples, true_params, observations, true_params_tensor = generate_true_parameters(
        prior=prior, posterior=posterior, vensimInputs=vensimInputs,
        SBI_Inputs=SBI_Inputs, lows=lows, highs=highs)

    if resInputs['graph_diagnostics']:
        # plot_posterior(
        #     posterior=posterior,
        #     lows=lows, highs=highs,
        #     vensimInputs=vensimInputs,
        #     resInputs=resInputs,
        #     x_o=x_o,
        #     true_parameter=true_parameter
        # )

        plot_ground_truth(
            post_samples=posterior_samples,
            true_params=true_params,
            param_names=list(vensimInputs['venParameters'].keys()),
            resInputs=resInputs
        )

        plot_z_score(
            post_samples=posterior_samples,
            true_params=true_params,
            param_names=list(vensimInputs['venParameters'].keys()),
            resInputs=resInputs
        )

        plot_confidence_intervals(
            post_samples=posterior_samples,
            true_params=true_params,
            param_names=list(vensimInputs['venParameters'].keys()),
            resInputs=resInputs
        )

    resInputs['scenario_name'] = scenario_name

    return posterior, (posterior_samples, true_params, observations, true_params_tensor)

def run_training(vensimInputs, SBI_Inputs, resInputs):
    """
    Run the training of a neural network, with the option of calculating validation statistics

    -----------
    Parameters:
        - vensimInputs (dictionary): customizable settings pertaining to simulating
        - SBI_Inputs (dictionary): customizable settings pertaining to training
        - res_Inputs (dictionary): customizable settings pertaining to plotting

    Returns:
        (Posterior) learned posterior object    
    """

    #generate one prior for all calculations; do not make new prior for validation
    vensimInputs['venParameters'] = parse_voc_file(vensimInputs)
    lows = torch.tensor([val[0] for val in vensimInputs['venParameters'].values()], dtype=torch.float32, device=vensimInputs['device'])
    highs = torch.tensor([val[1] for val in vensimInputs['venParameters'].values()], dtype=torch.float32, device=vensimInputs['device'])
    means = (lows + highs) / 2
    std_dev = (highs - lows) / 4
    prior = MultivariateNormal(loc=means, covariance_matrix=torch.diag(std_dev**2))
    
    if resInputs['calculate_validation_metrics']: # going to graph validation metrics, run multiple times
        prior_range = (highs - lows).cpu().numpy() # element-wise subtraction
        prior_vars = (prior_range ** 2)/12 # variance of each parameter, formula (b-a)^2/12
        prior_std_dev = refit_prior_bounds(prior.sample((10000,)), lows, highs).std(dim=0).cpu()

        # these are actually pretty close to each other, which is a good thing!
        #print("std_dev: ", std_dev)
        #print("prior_std_dev: ", prior_std_dev)
        
        def run_training_wrapper():
            valid_calc = {
            'rmse' : np.array([]),
            'agg_post_cont' : np.array([]),
            'agg_int_score' : np.array([]),
            'coverage_probabilities' : {cl: np.array([]) for cl in resInputs['credible_levels']},
            'overall_coverage' : np.array([]),
            'agg_z_score' : np.array([]),
            'z_score_std' : np.array([]),
            'sbc_ks_pvals' : np.array([]),
            }

            for sample_size in SBI_Inputs['multiple_sample_sizes']:
                SBI_Inputs['sampleSize'] = sample_size
                posterior, (posterior_samples, true_params, observations, true_params_tensor) = _train(prior, vensimInputs, SBI_Inputs, resInputs, lows, highs)
                valid_calc['rmse'] = np.append(valid_calc['rmse'], calculate_rmse(posterior_samples, true_params, prior_range, prior_std_dev))
                valid_calc['agg_post_cont'] = np.append(valid_calc['agg_post_cont'], calculate_agg_post_cont(posterior_samples, prior_vars))
                valid_calc['agg_int_score'] = np.append(valid_calc['agg_int_score'], calculate_agg_int_score(posterior_samples, true_params, prior_std_dev,
                                                                                                             credible_levels=resInputs['credible_levels'],
                                                                                                             valid_size=SBI_Inputs['valid_size']))
                #print("agg_int_score:", valid_calc['agg_int_score'])
                # Unpack overall calibration score and individual coverage scores.
                overall_coverage, coverage_scores = calculate_coverage_prob(
                    posterior_samples, true_params,
                    credible_levels=resInputs['credible_levels'],
                    valid_size=SBI_Inputs['valid_size']
                )

                for index, coverage_level in enumerate(resInputs['credible_levels']):
                    valid_calc['coverage_probabilities'][coverage_level] = np.append(valid_calc['coverage_probabilities'][coverage_level], coverage_scores[index])
                valid_calc['overall_coverage'] = np.append(valid_calc['overall_coverage'], overall_coverage)
                
                #valid_calc['agg_z_score'], valid_calc['z_score_std'] = np.append(valid_calc['agg_z_score'], calculate_agg_z_score(posterior_samples, true_params))
                agg_z, z_std = calculate_agg_z_score(posterior_samples, true_params)
                valid_calc['agg_z_score'] = np.append(valid_calc['agg_z_score'], agg_z)
                valid_calc['z_score_std'] = np.append(valid_calc['z_score_std'], z_std)
                #valid_calc['sbc_ks_pvals'] = np.append(valid_calc['sbc_ks_pvals'], calculate_sbc(posterior, observations.squeeze(dim=1), true_params_tensor,
                #                                                    list(vensimInputs['venParameters'].keys()), SBI_Inputs, resInputs, sample_size))
            print(f'valid_calc: {valid_calc}')

            # print("Final agg_int_score array:", valid_calc['agg_int_score'])
            # print("Max agg_int_score:", np.max(valid_calc['agg_int_score']))
            # print("Min agg_int_score:", np.min(valid_calc['agg_int_score']))
            # print("Mean agg_int_score:", np.mean(valid_calc['agg_int_score']))

            # max_score = np.max(valid_calc['agg_int_score']) # we're not going to normalize the interval as the range is informative

            # valid_calc['agg_int_score'] = valid_calc['agg_int_score'] / max_score

            return valid_calc, posterior
        
        if SBI_Inputs['amortized'] == False: # doing sequential training
            print('running sequential')
            if vensimInputs['true_parameter_file'] is not None: # we have interesting_true_parameters
                print('we have true params file')
                import importlib.util
                file_path = f"setup_functions/true_params/{vensimInputs['true_parameter_file']}"
                spec = importlib.util.spec_from_file_location("tensor_module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                interesting_true_parameters = module.true_params_tensor.to(vensimInputs['device'])
            else:
                interesting_true_parameters = generate_priors(vensimInputs['num_true_parameters'], vensimInputs['venParameters'])[0].to(vensimInputs['device'])

            array_size = len(SBI_Inputs['multiple_sample_sizes'])
            cumulative_validation_calculations = {
                'rmse' : np.zeros(array_size),
                'agg_post_cont' : np.zeros(array_size),
                'agg_int_score' : np.zeros(array_size),
                'coverage_probabilities' : {cl: np.zeros(array_size) for cl in resInputs['credible_levels']},
                'overall_coverage' : np.zeros(array_size),
                'agg_z_score' : np.zeros(array_size),
                'z_score_std' : np.zeros(array_size),
                'sbc_ks_pvals' : np.zeros(array_size),
            }
            for parameter in interesting_true_parameters:
                vensimInputs['true_parameter'] = parameter
                vensimInputs['x_o'] = None
                print(f'parameter: {parameter}')
                valid_calc, posterior = run_training_wrapper()
                for key, calculation_array in valid_calc.items(): # add the new metrics to cumulative_validation_calculations
                    if key == 'coverage_probabilities':
                        for cl, _ in calculation_array.items():
                            cumulative_validation_calculations[key][cl] = np.vstack(cumulative_validation_calculations[key][cl], valid_calc[key][cl])
                    else:
                        cumulative_validation_calculations[key] = np.vstack(cumulative_validation_calculations[key], valid_calc[key])
            aggregate_validation_calculations = {key: np.mean(value, axis=0) for key, value in cumulative_validation_calculations.items()}
        else: # running amortized
            print('running amortized')
            vensimInputs['true_parameter'] = None
            aggregate_validation_calculations, posterior = run_training_wrapper()

        indices = range(len(SBI_Inputs['multiple_sample_sizes']))
        sample_size_labels = [str(size) for size in SBI_Inputs['multiple_sample_sizes']]
        plot_rmse_results(indices, aggregate_validation_calculations['rmse'], sample_size_labels, resInputs)
        plot_agg_post_cont_results(indices, aggregate_validation_calculations['agg_post_cont'], sample_size_labels, resInputs)
        plot_agg_int_score_results(indices, aggregate_validation_calculations['agg_int_score'], sample_size_labels, resInputs)
        #plot_dimension_wise_fraction_within_interval(indices, resInputs['credible_levels'], aggregate_validation_calculations['coverage_probabilities'], sample_size_labels, resInputs)
        plot_overall_fraction_within_interval(indices, aggregate_validation_calculations['overall_coverage'], sample_size_labels, resInputs)
        #plot_zscore_results(indices, aggregate_validation_calculations['agg_z_score'], sample_size_labels, resInputs)
        #plot_zscore_std_results(indices, aggregate_validation_calculations['z_score_std'], sample_size_labels, resInputs)
        plot_zscore_merged_results(indices, aggregate_validation_calculations['agg_z_score'], aggregate_validation_calculations['z_score_std'], sample_size_labels, resInputs)
        #plot_sbc_results(indices, aggregate_validation_calculations['sbc_ks_pvals'], sample_size_labels, resInputs)
    
    
    else:
        posterior, _ = _train(prior, vensimInputs, SBI_Inputs, resInputs, lows, highs)

    return posterior
