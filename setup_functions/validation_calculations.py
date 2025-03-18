# validation_calculations.py
import numpy as np
import torch
from sbi.diagnostics.sbc import run_sbc, check_sbc
import os

np.random.seed(42)
torch.manual_seed(42)

def calculate_rmse(posterior_samples, true_params, prior_range, prior_std_dev):
    #print("true_params: ", true_params)
    post_means=np.mean(posterior_samples, axis=1)
    rmse_per_param = np.sqrt(np.mean((post_means - true_params) ** 2, axis=0))
    #print("rmse_per_param: ", rmse_per_param)
    #print("prior range:", prior_range)
    #print("prior standard deviation:", prior_std_dev)

    prior_std_dev_np = prior_std_dev.detach().cpu().numpy()
    #print("prior_std_dev_np: ", prior_std_dev_np)
    avg_agg_rmse = np.mean(rmse_per_param / prior_std_dev_np)
    #print("Average Aggregate RMSE: ", avg_agg_rmse)
    return avg_agg_rmse


def calculate_agg_post_cont(posterior_samples, prior_vars):
    # Compute the posterior means and variances across samples for each parameter
    post_vars = np.var(posterior_samples, axis=1, ddof=1)
    #print("post_vars: ", post_vars)
    post_contraction = 1 - post_vars/prior_vars
    #print("post_contraction per param: ", post_contraction)
    avg_post_contraction = np.mean(post_contraction)
    #print("avg post contraction: ", avg_post_contraction)
    return avg_post_contraction

def calculate_agg_int_score(posterior_samples, true_params, prior_std, credible_levels, valid_size):
    post_means = np.mean(posterior_samples, axis=1)
    all_interval_scores = []

    # Loop over credible levels to compute coverage probabilities
    for credible_level in credible_levels:
        lower_percentile = (1 - credible_level) / 2 * 100
        upper_percentile = (1 + credible_level) / 2 * 100
        interval_scores = []

        for i in range(valid_size):
            for j in range(post_means.shape[1]):
                samples = posterior_samples[i, :, j]
                y = true_params[i, j]

                L = np.percentile(samples, lower_percentile)
                U = np.percentile(samples, upper_percentile)

                penalty = 0
                if y < L:
                    penalty = (2 / (1 - credible_level)) * (L - y)
                elif y > U:
                    penalty = (2 / (1 - credible_level)) * (y - U)

                S = ((U - L) + penalty) / prior_std[j] # normalize by standard deviation

                interval_scores.append(S)
                #print("interval scores per param per credible: ", interval_scores)

        all_interval_scores.append(np.mean(interval_scores))
        #print("interval scores averaged on credible levels: ", all_interval_scores)

    mean_score = np.mean(all_interval_scores)
    #print("interval score averaged for the data size: ", mean_score)

    return mean_score


def calculate_coverage_prob(posterior_samples, true_params, credible_levels, valid_size):
    """
    Calculates a symmetric, normalized coverage probability score for each credible level.

    For each credible level, this function computes the empirical coverage (i.e., the fraction
    of ground truth values falling within the corresponding credible interval) and then compares it
    to the nominal credible level using a symmetric ratio:
    
        score = min(empirical_coverage / credible_level, credible_level / empirical_coverage)
    
    This formulation ensures that overcoverage and undercoverage are penalized equally:
      - A perfect match (empirical_coverage == credible_level) yields a score of 1.
      - Any deviation from the nominal level produces a score below 1.
      
    The overall score is computed as the mean of these normalized scores across all credible levels.

    Args:
        posterior_samples (np.ndarray): Array of shape (valid_size, num_samples, n_params)
            containing posterior samples for each parameter.
        true_params (np.ndarray): Array of shape (valid_size, n_params) with the ground truth parameters.
        credible_levels (list or array): List of credible levels (e.g., [0.50, 0.75, 0.95]).
        valid_size (int): Number of validation cases.

    Returns:
        overall_score (float): Mean normalized score across all credible levels.
        scores (list): List of normalized scores (one per credible level).
    """
    #print("true_params: ", true_params)
    n_params = true_params.shape[1]
    total_count = valid_size * n_params  # Total number of ground truth parameters
    scores = []

    for credible_level in credible_levels:
        lower_percentile = (1 - credible_level) / 2 * 100
        upper_percentile = (1 + credible_level) / 2 * 100

        count = 0
        for i in range(valid_size):
            for j in range(n_params):
                samples = posterior_samples[i, :, j]
                lower_bound = np.percentile(samples, lower_percentile)
                upper_bound = np.percentile(samples, upper_percentile)

                # if (j==2):
                #     print("lower_bound: ", lower_bound)
                #     print("true_params: ", true_params[i,j])
                #     #print("j: ", j)
                #     print("upper_bound: ", upper_bound)


                if lower_bound <= true_params[i, j] <= upper_bound:
                    count += 1

        empirical_coverage = count / total_count

        #print("empirical_coverage: ", empirical_coverage)

        # avoid division by 0, consider 1 to be a good score 
        if empirical_coverage == 0:
            relative_error = 1
        else:
            # we can remove/add the weight  
            # relative_error = abs(empirical_coverage - credible_level)/(1-credible_level)
            # relative_error = abs(empirical_coverage - credible_level) #check with Hazhir which version we should use
            relative_error = abs(empirical_coverage - credible_level)

        scores.append(relative_error)
        #print("scores for each credible: ", scores)

    # i think we were counting the weights twice (once in the original calculation and once in the overall_score)
    #overall_score = np.average(scores, weights=credible_levels)
    overall_score = np.average(scores)
    #print("average score across CIs for each training data : ", overall_score)
    return overall_score, scores



# def calculate_coverage_prob(posterior_samples, true_params, credible_levels, valid_size):
#     """
#     Calculates a symmetric, normalized coverage probability score for each credible level, based on parameter-wise coverage.

#     For each credible level, for each parameter, the function computes the fraction of cases in which the ground truth
#     falls within the credible interval (coverage for that parameter) and then calculates a relative error:

#         relative_error_per_param = |coverage - credible_level|   (if coverage is 0, it is set to 1)

#     For each credible level, the average relative error across parameters is computed and stored in scores.
#     In addition, the function prints the relative error per parameter for each credible level, and
#     before returning, it prints the aggregated relative error per parameter (averaged across all credible levels).

#     Args:
#         posterior_samples (np.ndarray): Array of shape (valid_size, num_samples, n_params)
#             containing posterior samples for each parameter.
#         true_params (np.ndarray): Array of shape (valid_size, n_params) with the ground truth parameters.
#         credible_levels (list or array): List of credible levels (e.g., [0.50, 0.75, 0.95]).
#         valid_size (int): Number of validation cases.

#     Returns:
#         overall_score (float): Mean normalized score across all credible levels.
#         scores (list): List of normalized scores (one per credible level).
#     """
#     import numpy as np
    
#     n_params = true_params.shape[1]
#     scores = []  # Average relative error per credible level
#     # To store relative error per parameter for each credible level
#     relative_errors_per_level = []
    
#     for credible_level in credible_levels:
#         lower_percentile = (1 - credible_level) / 2 * 100
#         upper_percentile = (1 + credible_level) / 2 * 100
        
#         # Compute relative error for each parameter separately
#         relative_error_per_param_list = []
#         for j in range(n_params):
#             count_for_param = 0
#             for i in range(valid_size):
#                 samples = posterior_samples[i, :, j]
#                 lower_bound = np.percentile(samples, lower_percentile)
#                 upper_bound = np.percentile(samples, upper_percentile)
                
#                 if lower_bound <= true_params[i, j] <= upper_bound:
#                     count_for_param += 1
            
#             # Coverage for parameter j
#             coverage_j = count_for_param / valid_size
            
#             # Compute relative error per parameter; if coverage is 0, assign error = 1
#             if coverage_j == 0:
#                 error = 1
#             else:
#                 error = abs(coverage_j - credible_level)
            
#             relative_error_per_param_list.append(error)
        
#         # Print relative error per parameter for this credible level
#         print(f"Credible Level {credible_level}: Relative Error per Param: {relative_error_per_param_list}")
        
#         # Average error for this credible level
#         avg_error_for_level = np.average(relative_error_per_param_list)
#         scores.append(avg_error_for_level)
        
#         # Save the per-parameter errors for later aggregation
#         relative_errors_per_level.append(relative_error_per_param_list)
    
#     # Aggregate relative error per parameter across credible levels
#     aggregated_relative_error_per_param = []
#     for j in range(n_params):
#         errors_for_param = [relative_errors_per_level[level_idx][j] for level_idx in range(len(credible_levels))]
#         aggregated_error = np.average(errors_for_param)
#         aggregated_relative_error_per_param.append(aggregated_error)
    
#     # Print the aggregated relative error per parameter
#     print(f"Aggregated Relative Error per Param (averaged across credible levels): {aggregated_relative_error_per_param}")
    
#     overall_score = np.average(scores)
#     return overall_score, scores




def calculate_agg_z_score(posterior_samples, true_params):
    """
    to be updated
    """
    #print("true_params: ", true_params)
    post_means = np.mean(posterior_samples, axis=1)  # shape: (valid_size, n_params)
    post_stds = np.std(posterior_samples, axis=1, ddof=1)  # shape: (valid_size, n_params)
    
    # Compute absolute z-scores for each parameter and validation case
    z_scores = (post_means - true_params) / (post_stds)
    #print("z_score per dimension: ", z_scores)

    # Aggregate by averaging over all cases and parameters
    agg_z_score_mean = np.mean(z_scores)

    #print("avg z_score across all dimensions per each dataset: ", agg_z_score_mean)

    z_scores_std = np.std(z_scores)
    #print("z_scores_std: ", z_scores_std)
    return agg_z_score_mean, z_scores_std


# def calculate_sbc(posterior, observations, true_params, param_names, SBI_Inputs):
#     # Use the fixed true parameters (e.g., 3 simulation runs) in place of prior_samples
#     param_ranks, param_dap_samples = run_sbc(
#         thetas = true_params,
#         xs = observations,
#         posterior = posterior,
#         num_posterior_samples=SBI_Inputs['post_samples_per_obs'],
#         reduce_fns='marginals',
#         num_workers=1 # this was previously set to 16
#     )
#     param_ks_pvals = check_sbc(param_ranks, true_params, param_dap_samples,
#                                SBI_Inputs['post_samples_per_obs'])['ks_pvals'].numpy()

#     print("param_ks_pvals: ", param_ks_pvals)
#     average_ks = np.mean(param_ks_pvals) # this is a naive average, we must use the threshold Hazhir said
#     #as long as it's not entirely fixed, let's print a warning
#     print("######### SBC is NOT finalized. It just prints a naive average for now! #########")
#     return average_ks

def calculate_sbc(posterior, observations, true_params, param_names, SBI_Inputs, resInputs, sample_size):
    # Use the fixed true parameters (e.g., 3 simulation runs) in place of prior_samples
    param_ranks, param_dap_samples = run_sbc(
        thetas = true_params,
        xs = observations,
        posterior = posterior,
        num_posterior_samples=SBI_Inputs['post_samples_per_obs'],
        reduce_fns='marginals',
        num_workers=1 # this was previously set to 16
    )
    param_ks_pvals = check_sbc(param_ranks, true_params, param_dap_samples,
                               SBI_Inputs['post_samples_per_obs'])['ks_pvals'].numpy()

    #print("param_ks_pvals: ", param_ks_pvals)
    threshold = 0.1
    count_above_threshold = np.sum(param_ks_pvals > threshold)
    fraction_above = count_above_threshold / len(param_ks_pvals)
    #print("Fraction of p-values above threshold {}: {}".format(threshold, fraction_above))
    
    # from sbi.analysis.plot import sbc_rank_plot
    # import matplotlib.pyplot as plt 
    # f, ax = sbc_rank_plot(
    #     ranks=param_ranks,
    #     num_posterior_samples=SBI_Inputs['post_samples_per_obs'],
    #     plot_type="hist",
    #     num_bins=20, # by passing None we use a heuristic for the number of bins.
    #     parameter_labels = param_names,
    #     num_cols = 5
    # )
    # directory = 'posterior'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig(f"{directory}/{resInputs['scenario_name']}_{sample_size}_sbc_rank_plot.svg", format='svg', dpi=300)
    # plt.show()

    return fraction_above
