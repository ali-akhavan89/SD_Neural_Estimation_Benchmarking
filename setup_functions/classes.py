"""
embedding network classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiConv1D(nn.Module):
    """
    Applies multiple 1-D convolutional layers to an input tensor
    """
    def __init__(self, in_channels, out_channels, min_kernel_size, max_kernel_size):
        """
        Initializes the MultiConv1D module with multiple 1D convolutional layers

        -----------
        Parameters:
            - in_channels (int): number of channels in the input tensor
            - out_channels (int): number of output channels for each convolutional layer
            - min_kernel_size (int): minimum kernel size for convolutional layer
            - max_kernel_size (int): maximum kernel size for convolutional layer
        """

        super(MultiConv1D, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                padding=ks // 2  # Ensure consistent output size with padding
            ) for ks in range(min_kernel_size, max_kernel_size + 1)
        ])

    def forward(self, x):
        """
        Applies the convolutional layers to the input tensor

        -----------
        Parameters:
            - x (tensor): input tensor

        Returns:
            (tensor) output tensor after applying the convolutional layers
        """
        
        conv_outputs = [conv(x) for conv in self.convs]
        min_length = min([output.shape[2] for output in conv_outputs])
        conv_outputs = [output[:, :, :min_length] for output in conv_outputs]
        x = torch.cat(conv_outputs, dim=1)
        return x


class SequenceNetwork(nn.Module):
    """
    Processes sequential data using multiple 1-D convolutional layers, followed by an LSTM and a fully connected layer.
    """
    def __init__(self, vensimInputs, summary_dim=10, num_conv_layers=3, lstm_units=128, bidirectional=True,
                 out_channels=32, min_kernel_size=1, max_kernel_size=3, initial_in_channels=1):
        """
        Initializes SequenceNetwork class

        -----------
        Parameters:
            - vensimInputs (dictionary): customizable settings pertaining to simulation
            - summary_dim (int): number of summary dimensions (default=10)
            - num_conv_layers (int): number of convolution layers (default=3)
            - lstm_units (int): number of lstm units (default=128)
            - bidirectional (bool): whether the network is bidirectional (default=True)
            - out_channels (int): number of output channels (default=32)
            - min_kernal_size (int): minimum kernel size (default=1)
            - max_kernel_size (int): maximum kernel size (default=3)
            - initial_in_channels (int): initial input channels (default=1)
        """

        super(SequenceNetwork, self).__init__()
        self.vensimInputs = vensimInputs
        self.conv_layers = nn.ModuleList()

        for _ in range(num_conv_layers):
            self.conv_layers.append(
                MultiConv1D(
                    in_channels=initial_in_channels,
                    out_channels=out_channels,
                    min_kernel_size=min_kernel_size,
                    max_kernel_size=max_kernel_size
                )
            )
            # Update in_channels for next layer to match concatenated output
            initial_in_channels = out_channels * (max_kernel_size - min_kernel_size + 1)

        self.lstm = nn.LSTM(
            input_size=initial_in_channels,
            hidden_size=lstm_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = lstm_units * 2 if bidirectional else lstm_units

        # **Define num_series here using vensimInputs**
        num_series = len(vensimInputs['outputs'])

        # Adjust output layer size if manual data is used
        if vensimInputs['manual_summaries']:
            manual_data_dim = vensimInputs['manual_dimensions'] * num_series
            self.out_layer = nn.Linear(lstm_output_size + manual_data_dim, summary_dim)
        else:
            self.out_layer = nn.Linear(lstm_output_size, summary_dim)

    def forward(self, x):
        """
        Processes input data through MultiConv1D, LSTM, and fully connected layers.

        -----------
        Parameters:
            - x (tensor): input ata

        Returns:
            (tensor) data passed through the sequence network
        """

        vensimInputs = self.vensimInputs

        if vensimInputs['manual_summaries']:
            time_series_length = len(vensimInputs['Time_points'])
            manual_summary_length = vensimInputs['manual_dimensions']
            x_time_series = x[:, :time_series_length, :]  # Shape: (batch_size, time_series_length, n_series)
            x_manual_summaries = x[:, time_series_length:, :]  # Shape: (batch_size, manual_summary_length, n_series)
        else:
            x_time_series = x  # No manual summaries
            x_manual_summaries = None

        x_time_series = x_time_series.permute(0, 2, 1)  # Shape: (batch_size, n_series, time_series_length)

        for conv_layer in self.conv_layers:
            x_time_series = conv_layer(x_time_series)

        x_time_series = x_time_series.permute(0, 2, 1)  # Shape: (batch_size, time_steps, conv_output_channels)
        self.lstm.flatten_parameters()  # Optimize LSTM performance on GPU
        x_processed, _ = self.lstm(x_time_series)
        x_processed = x_processed[:, -1, :]  # Take the last time step output

        if x_manual_summaries is not None:
            # Flatten x_manual_summaries
            x_manual_summaries_flat = x_manual_summaries.reshape(x_manual_summaries.size(0), -1)
            # Concatenate output and x_manual_summaries_flat
            x_combined = torch.cat((x_processed, x_manual_summaries_flat), dim=1)
            output = F.relu(self.out_layer(x_combined))
        else:
            output = F.relu(self.out_layer(x_processed))

        return output