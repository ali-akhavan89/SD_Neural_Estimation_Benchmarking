"""
functions run before start of model; used to prepare data and files, get necessary info
"""
import torch
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def parse_voc_file(vensimInputs):
    """
    parse a Vensim VOC (Varibale Output Configuration) file to extract parameter bounds

    -----------
    Parameters:
        - vensimInputs (dictionary): customizable settings pertaining to simulating
    
    Returns:
        (dictionary) dictionary, keys are parameter names, values are tuples of (lower_bound, upper_bound)
    """

    variables = {}
    with open(vensimInputs['venVOCFile'], 'r') as file:
        start_processing = False
        for line in file:
            if line.startswith(':'):
                start_processing = True
                continue

            if start_processing and line.strip():
                # Extract parts of the line
                parts = line.strip().split('<=')
                param = parts[1].strip()  # The parameter name is the second part
                lower_bound = float(parts[0].strip())
                upper_bound = float(parts[2].strip())
                variables[param] = (lower_bound, upper_bound)
    return variables


def create_tab_delimited(parameter_matrix, parameter_names, filename, seed_start, SeedVar='NSeed', numRow=1):
    """
    creates a tab-delimited file containing simulation values and a seed value

    -----------
    Parameters:
        - parameter_matrix (numpy.ndarray): 2D array, each row represents a set of parameters for a simulation
        - parameter_names (list): parameter names to be used as column headers in the file
        - filename (str): name of the file to save the tab-delimited data
        - seed_start (int): seed starting value; seed value is computed as `i + seed_start`, `i` is row index
        - SeedVar (str): name of the seed column in the output file (default='NSeed')
    """

    with open(filename, "w") as f:
        # Write the header row with all column names
        header = [SeedVar] + parameter_names
        f.write("\t".join(header) + "\n")

        # Write the parameter values, NSeed, and constants for each simulation
        for i in range(len(parameter_matrix)):
            for j in range(numRow):
                simulation_values = np.insert(parameter_matrix[i], 0, i * numRow + j + 1 + seed_start)  # Insert NSeed at the beginning
                f.write("\t".join(map(str, simulation_values)) + "\n")


def write_strings_to_file(strings, filename):
    """
    writes list of strings to a file; each string on new line

    -----------
    Parameters:
        - strings (list): strings to be written to the file
        - filename (str): name of the file to save the strings
    """

    with open(filename, "w") as f:
        for string in strings:
            f.write(f"{string}\n")  # Write each string with a newline character


def extract_sensitivities(model, variable_names, all_times, sensitivity_count, SeedVar='NSeed'):
    """
    extracts and sorts sensitivity data from a model for given variables and time points

    -----------
    Parameters:
        - model (object): model from which sensitivities are extracted; must have `getsensitivity` method
        - variable_names (list): variable names for which sensitivities are to be extracted
        - all_times (list): time points at which sensitivities are to be extracted
        - sensitivity_count (int): expected number of sensitivities (i.e., number of simulations)
        - SeedVar (str): seed values for sorting (default='NSeed')

    Returns:
        (numpy.ndarray) 3D array containing the sorted sensitivities.
    """

    # Extract the NSeed values for sorting
    vec, length = model.getsensitivity(SeedVar, 0)
    nseed_values = np.ctypeslib.as_array(vec, shape=(length,))
    
    # Check that the number of NSeed values matches the expected sensitivity count
    if length != sensitivity_count:
        raise ValueError(f"Expected sensitivity count {sensitivity_count}, but got {length}")
    
    # Sort the indices of simulations based on NSeed values
    sorted_indices = np.argsort(nseed_values)

    # Initialize an array to hold the sorted sensitivities
    sorted_sensitivities = np.empty((sensitivity_count, len(all_times), len(variable_names)))

    # Extract and sort data for each variable and time point
    for i, var_name in enumerate(variable_names):
        for j, time in enumerate(all_times):
            # Get the sensitivity vector for the current variable at the current time
            vec, _ = model.getsensitivity(var_name, time)
            sensitivities = np.ctypeslib.as_array(vec, shape=(sensitivity_count,))

            # Sort the sensitivities based on sorted_indices
            sorted_sensitivities[:, j, i] = sensitivities[sorted_indices]

    return sorted_sensitivities


def generate_summary_statistics(data, vensimInputs):
    """
    Generates summary statistics for time series data, including fitted values from the Savitzky-Golay filter,
    error terms, various moments, and cross-series correlations.

    -----------
    Parameters:
        - data (numpy.ndarray): 2D array to calculate statistics
        - vensimInputs (dictionary): customizable settings pertaining to simulating

    Returns:
        (np.array) summary statistics
    """

    if vensimInputs['window_length'] > data.shape[1] or vensimInputs['window_length'] <= vensimInputs['polyorder']:
        raise ValueError("Window length must be greater than polyorder and less than or equal to the number of time points.")

    num_series, num_times = data.shape
    all_stats = []  # List to store summary statistics per series
    fitted_lines = []
    fitted_errs = []

    for i in range(num_series):
        # Get fitted lines to data
        fitted_line = savgol_filter(data[i], vensimInputs['window_length'], vensimInputs['polyorder'])
        fitted_lines.append(fitted_line)

    # Error terms and their absolute values
    error_terms = data - np.array(fitted_lines)
    abs_errors = np.abs(error_terms)

    for i in range(num_series):
        # Get fitted absolute errors
        fitted_err = savgol_filter(
            abs_errors[i],
            min(vensimInputs['window_length'] * 2, abs_errors.shape[1]),
            vensimInputs['polyorder'] * 2
        )
        fitted_errs.append(fitted_err)

    for i in range(num_series):
        series_stats = []

        data_mean = np.mean(data[i])

        err_mean = np.mean(fitted_errs[i])

        series_stats.append(err_mean)
        series_stats.append(data_mean)

        if vensimInputs['num_moments'] >= 1:
            series_stats.append(np.mean(data[i]))
        if vensimInputs['num_moments'] >= 2:
            series_stats.append(np.std(data[i]))
        if vensimInputs['num_moments'] >= 3:
            series_stats.append(skew(data[i]))
        if vensimInputs['num_moments'] >= 4:
            series_stats.append(kurtosis(data[i]))

        # Savitzky-Golay filter fit values and error fit values
        fitted_line = fitted_lines[i]
        indices = np.linspace(0, data.shape[1] - 1, num=vensimInputs['num_points'] + 2, dtype=int)
        series_stats.extend(fitted_line[indices] / data_mean)

        fitted_err = fitted_errs[i]
        series_stats.extend(fitted_err[indices] / err_mean)

        # Cross-series correlations
        for lag in vensimInputs['lags']:
            for j in range(num_series):
                if lag >= 0:
                    if lag > 0:
                        vector1 = error_terms[i, :-lag]
                        vector2 = error_terms[j, lag:]
                    else:
                        vector1 = error_terms[i, :]
                        vector2 = error_terms[j, :]
                else:
                    vector1 = error_terms[i, :lag]
                    vector2 = error_terms[j, -lag:]
                # Handle cases where the vectors might be empty
                if vector1.size > 1 and vector2.size > 1:
                    corr_value = np.corrcoef(vector1, vector2)[0, 1]
                else:
                    corr_value = 0.0  # Default value if insufficient data
                series_stats.append(corr_value)

        # Append series_stats to all_stats
        all_stats.append(series_stats)

        if vensimInputs['plot']:
            plt.figure()
            plt.plot(data[i], label='Original data')
            plt.plot(fitted_lines[i], label='Savitzky-Golay filter')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Series {i + 1}')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(abs_errors[i], label='Absolute Error data')
            plt.plot(fitted_errs[i], label='Savitzky-Golay filter')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Series {i + 1}')
            plt.legend()
            plt.show()

    # Convert list to NumPy array
    all_stats_array = np.array(all_stats)  # Shape: (num_series, num_summaries_per_series)
    vensimInputs['manual_dimensions'] = all_stats_array.shape[1]
    return all_stats_array
