import pickle 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_data(batch_x_og, batch_y, batch_x_mark, outputs, idx): 
    batch_x_shape = batch_x_og.shape
    # Extract the relevant data for the given index
    batch_x = batch_x_og[idx, :, -1]
    batch_y = batch_y[idx]
    batch_x_mark = batch_x_mark[idx]
    outputs = outputs[idx]

    # Create x-axis indices for each series
    x_indices = range(len(batch_x))
    y_indices = range(len(batch_x), len(batch_x) + len(batch_y))
    outputs_indices = range(len(batch_x), len(batch_x) + len(outputs))

    # Plot the data
    plt.figure(figsize=(50, 10))
    plt.plot(x_indices, batch_x, color='blue', label='batch_x')
    plt.plot(y_indices, batch_y, color='red', label='batch_y')  # Plot batch_y after batch_x
    plt.plot(outputs_indices, outputs, color='orange', label='outputs')  # Plot outputs after batch_x
    # Uncomment the following line if batch_x_mark needs to be visualized
    # for i in range(batch_x_shape[2]):
    #     plt.plot(x_indices, batch_x_og[idx, :, i], label=f'batch_x_{i}')
    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    