"""
functions to plot diagnostic plots
"""

import torch
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score
from scipy.stats import median_abs_deviation
from sbi import analysis
import textwrap
from matplotlib.ticker import LogFormatterMathtext
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

# def plot_time_series(x_o, vensimInputs, resInputs):
#     time_points = vensimInputs['Time_points']
#     x_o_np = x_o.cpu().numpy() if hasattr(x_o, 'cpu') else x_o

#     feature_names = vensimInputs['feature_names'] if vensimInputs['feature_names'] else None
#     ylabels = vensimInputs['ylabels'] if vensimInputs['ylabels'] else [''] * len(feature_names)

#     num_cols = min(len(feature_names), 2)
#     num_rows = int(np.ceil(len(feature_names) / num_cols))

#     # it would throw an error if we don't wrap axes in an array
#     _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4))
#     if not hasattr(axs, "flatten"):
#         axs = np.array([axs])
#     else:
#         axs = axs.flatten()
    
#     for i in range(len(feature_names)):
#         axs[i].plot(time_points, x_o_np[0, :vensimInputs['constant_vals'][0]+1, i])
#         axs[i].set_title(feature_names[i])
#         axs[i].set_xlabel("Time (Month)")
#         axs[i].set_ylabel(ylabels[i])
    
    
#     plt.tight_layout()
#     if resInputs['save_graphs']:
#         directory = 'posterior'
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         plt.savefig(f"{directory}/{resInputs['scenario_name']}_time_series.svg", format='svg', dpi=300)
#     plt.show()


def plot_time_series(x_o, vensimInputs, resInputs, title_font_size=12, label_font_size=10, tick_font_size=10):
    time_points = vensimInputs['Time_points']
    x_o_np = x_o.cpu().numpy() if hasattr(x_o, 'cpu') else x_o

    feature_names = vensimInputs['feature_names'] if vensimInputs['feature_names'] else None
    ylabels = vensimInputs['ylabels'] if vensimInputs['ylabels'] else [''] * len(feature_names)

    num_cols = min(len(feature_names), 2)
    num_rows = int(np.ceil(len(feature_names) / num_cols))

    # it would throw an error if we don't wrap axes in an array
    _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4))
    if not hasattr(axs, "flatten"):
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    
    for i in range(len(feature_names)):
        #axs[i].plot(time_points, x_o_np[0, :vensimInputs['constant_vals'][0]+1, i], color='black', linestyle='--', marker='o', markevery=10, alpha=0.7)
        axs[i].plot(time_points, x_o_np[0, :vensimInputs['constant_vals'][0]+1, i], color='black', linestyle='--', linewidth=0.5)
        axs[i].set_title(feature_names[i], fontsize=title_font_size)
        axs[i].set_xlabel("Time (Month)", fontsize=label_font_size)
        axs[i].set_ylabel(ylabels[i], fontsize=label_font_size)
        axs[i].tick_params(axis='both', labelsize=tick_font_size)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].grid(True, color='lightgray', linewidth=0.5, alpha=0.7)
        axs[i].set_xlim(left=-0.05)
        axs[i].set_ylim(bottom=-0.05)
    
    plt.tight_layout()
    if resInputs['save_graphs']:
        directory = 'posterior'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{resInputs['scenario_name']}_time_series.svg", format='svg', dpi=300)
    plt.show()

# def plot_posterior(posterior, lows, highs, vensimInputs, resInputs, x_o, true_parameter):
#     """
#     plots posterior distribution of parameters given a simulation model

#     -----------
#     Parameters:
#         - posterior (sbi.inference.Posteriors): posterior distribution object from which samples are drawn
#         - lows (list): low value for each parameter of model
#         - highs (list): high value for each parameter of model
#         - vensimInputs (dictionary): customizable settings pertaining to simulating
#         - res_Inputs (dictionary): customizable settings pertaining to plotting
#         - x_o (tensor): one time point of simulated data for plotting
#         - true_parameter (numpy.ndarray): one random draw from the prior
#     """

#     posterior_samples_cpu = posterior.sample((resInputs['pairplot_sample_size'],), x=x_o).cpu().numpy()

#     wrapped_labels = [textwrap.fill(label, width=10) for label in vensimInputs['venParameters'].keys()]
    
#     # Plot posterior samples with true parameters marked
#     fig = analysis.pairplot(posterior_samples_cpu, 
#                             points=true_parameter.cpu().numpy().reshape(1, -1),
#                             labels=wrapped_labels,
#                             limits=[list(pair) for pair in zip(lows, highs)],
#                             points_colors="r",
#                             points_offdiag={"markersize": 6},
#                             figsize=(15, 15),
#                             #upper = "kde"
#                            )

#     if resInputs['save_graphs']:
#         directory = "posterior"
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         fig[0].savefig(directory+"/"+resInputs['scenario_name']+'_posterior.svg', format='svg', dpi=300)
#     if resInputs['show_graphs']:
#         plt.show()

def plot_posterior(posterior, lows, highs, vensimInputs, resInputs, x_o, true_parameter):
    """
    Plots the posterior distribution of parameters with a pairwise grid that resembles the one generated by plot_posterior_2d_GT.
    
    Parameters:
        - posterior (sbi.inference.Posteriors): the posterior distribution object from which samples are drawn
        - lows (list): the low value for each parameter (prior lower bound)
        - highs (list): the high value for each parameter (prior upper bound)
        - vensimInputs (dict): settings related to simulation; the keys of vensimInputs['venParameters'] serve as parameter names
        - resInputs (dict): settings related to plotting; keys include 'pairplot_sample_size', 'save_graphs', 'scenario_name', and 'show_graphs'
        - x_o (tensor): a single time point of simulated data used for plotting
        - true_parameter (numpy.ndarray or torch.Tensor): one draw from the prior that represents the true parameter values
    """
    # Sample from the posterior and convert the result to a NumPy array
    posterior_samples_cpu = posterior.sample((resInputs['pairplot_sample_size'],), x=x_o).cpu().numpy()
    
    # Create labels for the parameters by wrapping the text at a width of 10 characters
    wrapped_labels = [textwrap.fill(label, width=10) for label in vensimInputs['venParameters'].keys()]
    
    # Create a DataFrame from the posterior samples using the wrapped parameter labels
    posterior_draws_df = pd.DataFrame(posterior_samples_cpu, columns=wrapped_labels)
    
    # Plot settings
    post_color = "#8f2727"
    post_alpha = 0.9
    height = 3
    label_fontsize = 14
    tick_fontsize = 12

    # Create a pair grid from the DataFrame
    g = sns.PairGrid(posterior_draws_df, height=height)
    g.map_diag(sns.histplot, fill=True, color=post_color, alpha=post_alpha, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)
    
    # Convert true_parameter to CPU and a NumPy array if necessary, then add markers for the true parameter values
    if true_parameter is not None:
        if hasattr(true_parameter, 'cpu'):
            true_parameter = true_parameter.cpu().numpy()
        n_params = len(true_parameter)
        for i, val in enumerate(true_parameter):
            g.axes[i, i].axvline(val, color='k', linestyle='--')
            for j in range(i):
                g.axes[i, j].plot(true_parameter[j], val, 'ko')
            for j in range(i + 1, n_params):
                g.axes[j, i].plot(val, true_parameter[j], 'ko')
    
    # Turn off the axes for the upper triangle
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis("off")
    
    # Adjust tick parameters for the lower triangle
    for i, j in zip(*np.tril_indices_from(g.axes, 1)):
        g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)
    
    # Set the labels for the y-axis on the left column and for the x-axis on the bottom row
    for i, param_name in enumerate(wrapped_labels):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(wrapped_labels) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)
    
    # Add grid lines to every subplot
    n_params = len(wrapped_labels)
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)
    
    # Set the x- and y-axis limits using the prior boundaries.
    # Convert to CPU scalars if the boundaries are torch tensors.
    for i in range(n_params):
        for j in range(n_params):
            ax = g.axes[i, j]
            low_j = lows[j].cpu().item() if hasattr(lows[j], 'cpu') else lows[j]
            high_j = highs[j].cpu().item() if hasattr(highs[j], 'cpu') else highs[j]
            low_i = lows[i].cpu().item() if hasattr(lows[i], 'cpu') else lows[i]
            high_i = highs[i].cpu().item() if hasattr(highs[i], 'cpu') else highs[i]
            ax.set_xlim(low_j, high_j)
            ax.set_ylim(low_i, high_i)
    
    g.tight_layout()
    
    # Save the figure if required by the settings
    if resInputs['save_graphs']:
        directory = "posterior"
        if not os.path.exists(directory):
            os.makedirs(directory)
        g.fig.savefig(os.path.join(directory, f"{resInputs['scenario_name']}_posterior.svg"), format='svg', dpi=300)
    
    # Display the plot if required by the settings
    if resInputs['show_graphs']:
        plt.show()


# def plot_ground_truth(post_samples, true_params, resInputs, param_names=None, point_agg=np.median, uncertainty_agg=median_abs_deviation,
#                                add_r2=True, add_corr=True):
#     """
#     plots the ground truth values against the estimated parameter values

#     -----------
#     Parameters:
#         - post_samples (numpy.ndarray): posterior samples of the parameters
#         - true_params (numpy.ndarray): ground truth values of the parameters
#         - resInputs (dictionary): customizable settings pertaining to plotting
#         - param_names (list): parameter names for labeling plots (default=None)
#         - point_agg (function): aggregate the posterior samples into a point estimate (default=np.median)
#         - uncertainty_agg (function): aggregate the posterior samples into an uncertainty measure (default=median_abs_deviation)
#         - add_r2 (bool): adds the R-squared metric to the plot (default=True)
#         - add_corr (bool): adds the correlation coefficient to the plot (default=True)
#     """

#     est = point_agg(post_samples, axis=1)
#     if uncertainty_agg is not None:
#         u = uncertainty_agg(post_samples, axis=1)
    
#     n_params = est.shape[1]  # Use the shape of `est` to determine the number of parameters
#     if param_names is None:
#         param_names = [f"$\\theta_{{{i}}}$" for i in range(n_params)]

#     num_col = min(true_params.shape[1], 4)
#     num_row = int(np.ceil(true_params.shape[1]/num_col))
#     fig, axes = plt.subplots(num_row, num_col, figsize=(int(4 * num_col), int(4 * num_row)))
#     axes = np.atleast_1d(axes)
    
#     for index, axis in enumerate(([axes] if num_row == 1 else axes)):
#         for j, ax in enumerate(axis):
#             i = index*num_col + j
#             if i >= n_params:
#                 break
#             ax.errorbar(true_params[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color='#8f2727')
#             lower = min(true_params[:, i].min(), est[:, i].min())
#             upper = max(true_params[:, i].max(), est[:, i].max())
#             ax.plot([lower, upper], [lower, upper], linestyle='--', color='black')
    
#             # Add optional metrics and title
#             if add_r2:
#                 r2 = r2_score(true_params[:, i], est[:, i])
#                 ax.text(
#                     0.1, 0.9,
#                     f"$R^2$ = {r2:.3f}",
#                     horizontalalignment="left",
#                     verticalalignment="center",
#                     transform=ax.transAxes,
#                     size=18,
#                 )
#             if add_corr:
#                 corr = np.corrcoef(true_params[:, i], est[:, i])[0, 1]
#                 ax.text(
#                     0.1, 0.8,
#                     f"$r$ = {corr:.3f}",
#                     horizontalalignment="left",
#                     verticalalignment="center",
#                     transform=ax.transAxes,
#                     size=18,
#                 )
#             ax.set_title(param_names[i], fontsize=20)
#             ax.set_xlabel("Ground truth", fontsize=18)
#             ax.set_ylabel("Estimated", fontsize=18)
#             ax.tick_params(axis='both', labelsize=14)
    
#     plt.tight_layout()
#     if resInputs['save_graphs']:
#         directory = 'ground_truth'
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         plt.savefig(directory+'/'+resInputs['scenario_name']+'_ground_truth.svg', format='svg', dpi=300)
#     if resInputs['show_graphs']:
#         plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

def centered_k_percentile_range(data, k, axis=None):
    lower = np.percentile(data, k, axis=axis)
    upper = np.percentile(data, 100 - k, axis=axis)
    #print("data: ", data)
    return (upper - lower) / 2

# def plot_ground_truth(post_samples, true_params, resInputs, param_names=None, point_agg=np.mean, uncertainty_agg=None, add_r2=True, add_corr=True):

#     est = point_agg(post_samples, axis=1)
#     if uncertainty_agg is None and 'rcvr_prcnl' in resInputs:
#         k = (100 - resInputs['rcvr_prcnl']) / 2
#         print("k: ", k)
#         uncertainty_agg = lambda data, axis=None: centered_k_percentile_range(data, k, axis=axis)
#         u = uncertainty_agg(post_samples, axis=1)
#     # if uncertainty_agg is not None:
#     #     u = uncertainty_agg(post_samples, axis=1)
#     else:
#         u = None
#         print("u is not None")

#     n_params = est.shape[1]  # Use the shape of `est` to determine the number of parameters
#     if param_names is None:
#         param_names = [f"$\\theta_{{{i}}}$" for i in range(n_params)]

#     num_col = min(true_params.shape[1], 5)
#     num_row = int(np.ceil(true_params.shape[1]/num_col))
#     fig, axes = plt.subplots(num_row, num_col, figsize=(int(4 * num_col), int(4 * num_row)))
#     axes = np.atleast_1d(axes)
    
#     for index, axis in enumerate(([axes] if num_row == 1 else axes)):
#         for j, ax in enumerate(axis):
#             i = index*num_col + j
#             if i >= n_params:
#                 break
#             if u is not None:
#                 ax.errorbar(true_params[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color='#8f2727')
#             else:
#                 ax.plot(true_params[:, i], est[:, i], "o", alpha=0.5, color='#8f2727')
#             lower = min(true_params[:, i].min(), est[:, i].min())
#             upper = max(true_params[:, i].max(), est[:, i].max())
#             eps = (upper - lower) * 0.1
#             ax.set_xlim([lower - eps, upper + eps])
#             ax.set_ylim([lower - eps, upper + eps])
#             ax.plot(
#                 [ax.get_xlim()[0], ax.get_xlim()[1]],
#                 [ax.get_ylim()[0], ax.get_ylim()[1]],
#                 color="black",
#                 alpha=0.9,
#                 linestyle="dashed",
#             )
    
#             # Add optional metrics and title
#             if add_r2:
#                 r2 = r2_score(true_params[:, i], est[:, i])
#                 ax.text(
#                     0.1, 0.9,
#                     f"$R^2$ = {r2:.3f}",
#                     horizontalalignment="left",
#                     verticalalignment="center",
#                     transform=ax.transAxes,
#                     size=18,
#                 )
#             if add_corr:
#                 corr = np.corrcoef(true_params[:, i], est[:, i])[0, 1]
#                 ax.text(
#                     0.1, 0.8,
#                     f"$r$ = {corr:.3f}",
#                     horizontalalignment="left",
#                     verticalalignment="center",
#                     transform=ax.transAxes,
#                     size=18,
#                 )
#             sns.despine(ax=ax)
#             ax.grid(alpha=0.5)
#             ax.set_title(param_names[i], fontsize=20)
#             if index == num_row - 1:
#                 ax.set_xlabel("Ground truth", fontsize=18)
#             ax.set_ylabel("Estimated", fontsize=18)
#             ax.tick_params(axis='both', labelsize=14)

#     # Hide any unused subplots
#     for ax in axes.flatten()[n_params:]:
#         ax.set_visible(False)
    
#     plt.tight_layout()
#     if resInputs['save_graphs']:
#         directory = 'ground_truth'
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         plt.savefig(directory+'/'+resInputs['scenario_name']+'_ground_truth.svg', format='svg', dpi=300)
#     if resInputs['show_graphs']:
#         plt.show()


def plot_ground_truth(post_samples, true_params, resInputs, param_names=None, point_agg=np.mean, uncertainty_agg=None, add_r2=True, add_corr=True):

    est = point_agg(post_samples, axis=1)
    #print("Fourth dimension estimation:", est[:, 3])
    #print("Fourth dimension ground truth:", true_params[:, 3])
    if uncertainty_agg is None and 'rcvr_prcnl' in resInputs:
        k = (100 - resInputs['rcvr_prcnl']) / 2
        #print("k: ", k)
        uncertainty_agg = lambda data, axis=None: centered_k_percentile_range(data, k, axis=axis)
        u = uncertainty_agg(post_samples, axis=1)
    else:
        u = None
        #print("u is not None")

    n_params = est.shape[1]  # Use the shape of `est` to determine the number of parameters
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(n_params)]

    label_fontsize = 16
    title_fontsize = 18
    metric_fontsize = 16
    tick_fontsize = 12
    xlabel = "Ground truth"
    ylabel = "Estimated"

    num_col = min(true_params.shape[1], 5)
    num_row = int(np.ceil(true_params.shape[1]/num_col))
    fig, axes = plt.subplots(num_row, num_col, figsize=(int(4 * num_col), int(4 * num_row)))
    # Flatten the axes array for consistent iteration
    axarr = np.atleast_1d(axes)
    if num_col > 1 or num_row > 1:
        axarr = axarr.flat
    for i, ax in enumerate(axarr):
        if i >= n_params:
            break
        if u is not None:
            ax.errorbar(true_params[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color='#8f2727')
        else:
            ax.scatter(true_params[:, i], est[:, i], alpha=0.5, color='#8f2727')
        lower = min(true_params[:, i].min(), est[:, i].min())
        upper = max(true_params[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )
        if add_r2:
            r2 = r2_score(true_params[:, i], est[:, i])
            ax.text(
                0.1, 0.9,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(true_params[:, i], est[:, i])[0, 1]
            ax.text(
                0.1, 0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.set_title(param_names[i], fontsize=title_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Set x-labels for the bottom row and y-labels for the left-most column
    axarr_2d = np.array(axes)
    if num_row == 1:
        axarr_2d[0].set_ylabel(ylabel, fontsize=label_fontsize)
        for ax in axarr_2d:
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
    else:
        for ax in (axarr_2d[num_row-1, :] if num_col > 1 else axarr_2d):
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
        for ax in axarr_2d[:, 0]:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Remove any unused subplots
    for ax in list(np.array(axes).flatten())[n_params:]:
        ax.remove()

    plt.tight_layout()
    if resInputs['save_graphs']:
        directory = 'ground_truth'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+'/'+resInputs['scenario_name']+'_ground_truth.svg', format='svg', dpi=300)
    if resInputs['show_graphs']:
        plt.show()


def plot_z_score(post_samples, true_params, resInputs, param_names=None, fig_size=None, label_fontsize=16, title_fontsize=18,
                 tick_fontsize=12, n_col=None, n_row=None):
    """
    plots posterior contraction versus posterior z-scores

    -----------
    Parameters:
        - post_samples (numpy.ndarray): posterior samples of the parameters
        - true_params (numpy.ndarray): ground truth values of the parameters
        - resInputs (dictionary): customizable settings pertaining to plotting
        - param_names (list): parameter names for labeling plots (default=None)
        - fig_size (tuple): size of the figure (default=None)
        - label_fontsize (int) font size for axis labels (default=16)
        - title_fontsize (int) font size for subplot titles (default=18)
        - tick_fontsize (int): font size for axis tick labels (default=12)
        - n_col (int): number of columns in the subplot grid (default=None)
        - n_row (int): number of rows in the subplot grid (default=None)
    """

    # Calculate posterior means and standard deviations for each parameter
    post_means = np.mean(post_samples, axis=1)
    post_stds = np.std(post_samples, axis=1, ddof=1)
    post_vars = np.var(post_samples, axis=1, ddof=1)

    # Calculate prior variances from true parameters (assumed to be prior samples)
    prior_vars = np.var(true_params, axis=0, ddof=1)

    # Compute contraction for each parameter
    post_cont = 1 - (post_vars / prior_vars)

    # Compute posterior z-scores
    # Each sample's mean is compared with the true parameter, normalized by the posterior std
    z_score = (post_means - true_params) / post_stds

    # Determine the number of parameters and param names if None are given
    n_params = true_params.shape[1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine the number of rows and columns for subplots
    if n_row is None and n_col is None:
        n_col = min(n_params, 5)
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize the figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    fig, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    axarr = np.atleast_1d(axarr)
    axarr_it = axarr.flat if (n_col > 1 or n_row > 1) else axarr

    # Plot the data
    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break
        # post_cont is averaged over samples; z_score needs to match post_samples in shape
        ax.scatter(post_cont[:, i], z_score[:, i], color='#8f2727', alpha=0.5)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-3, 3])

    # Add labels
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Posterior contraction", fontsize=label_fontsize)
    if n_row == 1:
        axarr[0].set_ylabel("Posterior z-score", fontsize=label_fontsize)
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel("Posterior z-score", fontsize=label_fontsize)

    # Remove unused axes
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    fig.tight_layout()
    if resInputs['save_graphs']:
        directory = 'z_score'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(directory+'/'+resInputs['scenario_name']+'_z_score.svg', format='svg', dpi=300)
    if resInputs['show_graphs']:
        plt.show()


def plot_confidence_intervals(post_samples, true_params, param_names, resInputs, N=20):
    """
    plots fraction of true parameter values falling within posterior confidence intervals

    -----------
    Parameters:
        - post_samples (numpy.ndarray): posterior samples of the parameters
        - true_params (numpy.ndarray): ground truth values of the parameters
        - param_names (list): parameter names for labeling the plots
        - resInputs (dictionary): customizable settings pertaining to plotting
        - N (int): number of confidence intervals to consider (default=20)
    """

    num_params = true_params.shape[1]
    num_columns = min(num_params, 5)
    num_rows = int(np.ceil(num_params / num_columns))
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 4, num_rows * 4))
    axes = axes.flatten() if num_params > 1 else [axes]

    for param_index in range(num_params):
        ax = axes[param_index]
        param_name = param_names[param_index]
        ground_truths = true_params[:, param_index]
        posteriors = post_samples[:, :, param_index]

        fractions = np.zeros(N)

        for i in range(1, N + 1):
            lower_percentile = 50 - 50 * i / N
            upper_percentile = 50 + 50 * i / N

            ci_lower = np.percentile(posteriors, lower_percentile, axis=1)
            ci_upper = np.percentile(posteriors, upper_percentile, axis=1)

            within_interval = (ground_truths >= ci_lower) & (ground_truths <= ci_upper)
            fractions[i - 1] = np.mean(within_interval)

        ax.plot(np.arange(1, N + 1) / N, fractions, marker='o', color='#8f2727', linestyle='-', alpha = 0.8)
        # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)  # 45-degree reference line
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )
        ax.set_title(f"{param_name}", fontsize=18)
        if (param_index // num_columns) == 1:
            ax.set_xlabel("Interval Size", fontsize=16)
        if param_index % num_columns == 0:
            ax.set_ylabel("Fraction Within Interval", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        #ax.grid(True)

    # Remove any extra axes
    for i in range(num_params, num_rows * num_columns):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if resInputs['save_graphs']:
        directory = 'confidence'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+'/'+resInputs['scenario_name']+'_confidence.svg', format='svg', dpi=300)
    if resInputs['show_graphs']:
        plt.show()


def plot_experiment_results(*results_dfs, x_key, y_key, x_label, y_label, df_labels, file_name, title_name):
    """
    plots results from experiments

    -----------
    Parameters:
        - *results_dfs (DataFrame): variable number of pandas DataFrames containing the experiment results
        - x_key (str): column name in the DataFrames for the x-axis values
        - y_key (str): column name in the DataFrames for the y-axis values
        - x_label (str): label for the x-axis
        - y_label (str): label for the y-axis
        - df_labels (list): labels for each DataFrame to be used in the plot legend
        - file_name (str): name of the file to save the plot as
        - title_name (str): title of the plot
    """

    if len(results_dfs) != len(df_labels):
        raise ValueError("The number of DataFrames and df_labels must be the same.")
    
    # Define markers and colors for differentiating between DataFrames
    markers = ['o', 's', '^', 'D', 'x', '>', '<', 'p', '*', '+']  # Extended with more marker types
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  # Extended with more colors

    
    plt.figure(figsize=(10, 6))
    
    for df_index, df in enumerate(results_dfs):
        if x_key not in df.columns or y_key not in df.columns:
            print(f"DataFrame {df_index} does not contain the specified keys.")
            continue
        
        x_values = df[x_key]
        y_values = df[y_key]
        labels = df['experiment_label']
        
        plt.scatter(x_values, y_values, label=df_labels[df_index], marker=markers[df_index % len(markers)], color=colors[df_index % len(colors)])
        plt.plot(x_values, y_values, color=colors[df_index % len(colors)])  # Connect points with a line
        
        # Annotate points with labels
        for (x, y, label) in zip(x_values, y_values, labels):
            plt.text(x, y, label, fontsize=9)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title_name)
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    plt.savefig('experiment_results/' + file_name + '.jpeg', format='jpeg',dpi=300)
    plt.show()

# original settings:

# ylabelfontsize = 22
# titlefontsize = 22
# xticksfontsize = 18
# yticksfontsize = 18

ylabelfontsize = 32
titlefontsize = 28
xticksfontsize = 26
yticksfontsize = 26
markersize = 10

def plot_rmse_results(indices, agg_rmse_list, sample_size_labels, resInputs):
    plt.figure()
    plt.plot(indices, agg_rmse_list, marker='o', markersize = markersize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.ylabel(resInputs['graph_yaxis'], fontsize = ylabelfontsize)
    plt.ylim(-0.05,1.55) # this will make the width of the horizontal lines equal :)
    plt.yticks(np.arange(0, 1.5 + 0.3, 0.3))
    plt.grid(True, axis='y', linestyle='-', linewidth = 0.7, alpha=0.7)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0e}'))
    #plt.xticks(indices, sample_size_labels, fontsize = 18)
    #plt.xticks(indices, [f'$10^{{{int(np.log10(int(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    plt.xticks([])
    plt.yticks(fontsize = yticksfontsize)
    plt.gca().tick_params(axis='both', which='both', length=0)
    plt.title('RMSE', fontsize = titlefontsize)
    plt.tight_layout() # let's put these explicitly inside the functions because sometimes I had issues with ylabel not showing up
    plt.savefig(f"posterior/{resInputs['scenario_name']}_rmse.svg", format='svg', dpi=300) # let's save the graphs in SVG
    plt.show()
    print(list(round(i, 2) for i in agg_rmse_list))

def plot_agg_post_cont_results(indices, agg_post_cont_list, sample_size_labels, resInputs):
    plt.figure()
    plt.plot(indices, agg_post_cont_list, marker='o', markersize = markersize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    #plt.ylabel(resInputs['graph_yaxis'], fontsize = ylabelfontsize)
    plt.ylim(-0.05,1.05)
    plt.grid(True, axis='y', color='darkgray', linewidth=0.7, linestyle='-', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0e}'))    
    #plt.xticks(indices, sample_size_labels)
    #plt.xticks(indices, [f'$10^{{{int(np.log10(int(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    plt.xticks([])
    plt.yticks(fontsize = yticksfontsize)
    plt.gca().tick_params(axis='both', which='both', length=0)
    plt.title('Posterior Contraction', fontsize = titlefontsize)
    plt.tight_layout()
    plt.savefig(f"posterior/{resInputs['scenario_name']}_post_cont.svg", format='svg', dpi=300)
    plt.show()
    print(list(round(i, 2) for i in agg_post_cont_list))

def plot_agg_int_score_results(indices, agg_interval_scores_list, sample_size_labels, resInputs):
    plt.figure()
    plt.plot(indices, agg_interval_scores_list, marker='o', markersize = markersize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    #plt.ylabel(resInputs['graph_yaxis'], fontsize = ylabelfontsize)
    plt.ylim(-0.05,4.05) # max 4 is risky because much more complex models like world dynamics might generate worse interval score
    plt.grid(True, axis='y', color='darkgray', linewidth=0.7, linestyle='-', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0e}'))    
    #plt.xticks(indices, sample_size_labels)
    #plt.yticks(fontsize = yticksfontsize)
    #plt.xticks(indices, [f'$10^{{{int(np.log10(int(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    plt.xticks([])
    plt.yticks(np.arange(0, 5, 1), fontsize=yticksfontsize)
    plt.gca().tick_params(axis='both', which='both', length=0)
    plt.title('Interval Score', fontsize = titlefontsize)
    plt.tight_layout()
    plt.savefig(f"posterior/{resInputs['scenario_name']}_int_score.svg", format='svg', dpi=300)
    plt.show()
    print(list(round(i, 2) for i in agg_interval_scores_list))

def plot_dimension_wise_fraction_within_interval(indices, credible_levels, coverage_probabilities_results, sample_size_labels, resInputs):
    # Plot Overall Coverage Probability vs Training Data Size for each credible level
    for credible_level in credible_levels:
        coverage_list = coverage_probabilities_results[credible_level]
        plt.figure()
        plt.plot(indices, coverage_list, marker='o')
        plt.xlabel('Training Data Size', fontsize = 22)
        plt.ylabel('Fraction within Interval')
        plt.title(f'Fraction within Interval at {int(credible_level*100)}% CI vs Training Data Size')
        plt.xticks(indices, sample_size_labels)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f"posterior/{resInputs['scenario_name']}_within_int_{credible_level}.svg", format='svg', dpi=300)
        plt.show()
        print(list(round(i, 2) for i in coverage_list))




def plot_overall_fraction_within_interval(indices, overall_coverage_results, sample_size_labels, resInputs):
    # Create a new figure and axis for clarity
    fig, ax = plt.subplots()

    # Plot the data with markers
    ax.plot(indices, overall_coverage_results, marker='o', markersize = markersize)

    # Hide all spines for a cleaner look
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Set the y-axis label
    #ax.set_ylabel(resInputs['graph_yaxis'], fontsize=ylabelfontsize)

    # Set the y-axis to a logarithmic scale and define its limits
    ax.set_yscale('log')
    ax.set_ylim(0.0005, 0.15)
    ax.minorticks_on()

    # Use a log formatter for the y-axis to ensure proper scientific notation
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Add a grid only on the y-axis for better readability
    ax.grid(True, which='both', axis='y', color='darkgray', linewidth=0.7, linestyle='-', alpha=0.7)

    # Format the x-axis ticks. Here, we assume that sample_size_labels are powers of ten
    #ax.set_xticks(indices)
    #ax.set_xticklabels([f'$10^{{{int(np.log10(float(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    # to disable the data sizes as labels on x-axis
    ax.tick_params(axis='x', labelbottom=False)
    
    # Remove tick marks for a minimalist style
    ax.tick_params(axis='both', which='both', length=0)
    plt.setp(ax.get_yticklabels(), fontsize=yticksfontsize)

    # Set the plot title
    ax.set_title('Coverage Fraction Error', fontsize=titlefontsize)

    # Adjust layout, save the figure and display it
    fig.tight_layout()
    fig.savefig(f"posterior/{resInputs['scenario_name']}_within_overall.svg", format='svg', dpi=300)
    plt.show()

    # Print rounded values of overall coverage
    print([round(i.item(), 2) for i in overall_coverage_results])




def plot_zscore_merged_results(indices, agg_zscore_results, agg_zscore_std_results, sample_size_labels, resInputs):
    fig, ax1 = plt.subplots()

    # Plot z-score mean on primary axis
    ax1.plot(indices, agg_zscore_results, marker='o', label='Mean', color='blue', markersize = markersize)
    ax1.plot(indices, agg_zscore_std_results, marker='o', label='Std', color='red', markersize = markersize)
    ax1.legend(loc='upper right', fontsize=16)
    #ax1.legend(fontsize=30)
    ax1.set_ylim(-1.05, 2.05)
    #ax1.set_ylabel(resInputs['graph_yaxis'], fontsize = ylabelfontsize)
    ax1.grid(True, axis='y', color='darkgray', linewidth=0.7, linestyle='-', alpha=0.7)
    # Remove spines for a cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0e}'))
    #ax1.set_xticks(indices)
    #ax1.set_xticklabels([f'$10^{{{int(np.log10(int(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    plt.xticks([])  
    plt.yticks(fontsize = yticksfontsize)
    ax1.tick_params(axis='both', which='both', length=0)

    # Combine legends from ax1
    #ax1.legend(loc='best')

    plt.title('z-Score Mean and Std Dev', fontsize=titlefontsize)
    plt.tight_layout()
    plt.savefig(f"posterior/{resInputs['scenario_name']}_z_score_merged.svg", format='svg', dpi=300)
    plt.show()

    print("z-Score Mean:", list(round(i, 2) for i in agg_zscore_results))
    print("z-Score Std:", list(round(i, 2) for i in agg_zscore_std_results))


def plot_sbc_results(indices, sbc_ks_pvals_results, sample_size_labels, resInputs):
    # Convert each element to a scalar (if they are PyTorch tensors)
    sbc_ks_pvals_results = [x.item() if hasattr(x, 'item') else float(x) for x in sbc_ks_pvals_results]
    
    plt.figure()
    plt.plot(indices, sbc_ks_pvals_results, marker='o', markersize = markersize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    #plt.ylabel(resInputs['graph_yaxis'], fontsize = ylabelfontsize)
    plt.ylim(-0.05,1.05)
    plt.grid(True, axis='y', color='darkgray', linewidth=0.7, linestyle='-', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0e}'))    
    #plt.xticks(indices, sample_size_labels)
    #plt.xticks(indices, [f'$10^{{{int(np.log10(int(label)))}}}$' for label in sample_size_labels], fontsize=xticksfontsize)
    plt.xticks([])  
    plt.yticks(fontsize = yticksfontsize)
    plt.gca().tick_params(axis='both', which='both', length=0)
    plt.title('Fraction KS p-values > 0.1', fontsize = titlefontsize)
    plt.tight_layout()
    plt.savefig(f"posterior/{resInputs['scenario_name']}_sbc.svg", format='svg', dpi=300)
    plt.show()
    print([round(x, 2) for x in sbc_ks_pvals_results])
