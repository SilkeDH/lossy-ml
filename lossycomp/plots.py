"""Plotting functions"""

import matplotlib.pyplot as plt

def plot_history(history, variable, title, x_title, y_title):
    """Plots training and validation loss curves"""
    loss = history.history[variable]
    val_loss = history.history['val_' + variable]
    epochs = range(1, len(loss) + 1)
    f = plt.figure(figsize=(10,10))
    plt.plot(epochs, loss, 'r-', label='Training ' + variable)
    plt.plot(epochs, val_loss, 'g', label='Validation ' + variable)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.show()

def single_plot(data, time, title):
    """Single lat-lon plot
    Args
    ======
    data: Data of size (batch, time, lat, lon, level).
    time: Time index to be sliced.
    title: Sets plot title.
    """
    fig, ax = plt.subplots()
    bb = plt.imshow(data[0,time,:,:,0],  cmap='viridis')
    fig.colorbar(bb, orientation='vertical')
    plt.title(title)
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    plt.show()
    
def mult_plot(data, rows, columns, title, min, max, cmap = 'viridis'):
    """Multiple lat-lon plot
    Args
    ======
    data: Data of size (batch, time, lat, lon, level).
    columns: # columns displayed.
    rows: # of rows displayed.
    title: Sets plot title.
    min: Min value in data.
    max: Max value in data.
    cmap: Colorin type of the plot.
    The time index go from 0 to rows*columns.
    """
    fig, ax = plt.subplots(rows, columns, figsize=(10,10))
    count = 0
    for i in range(rows):
        for j in range(columns):
            img = data[0,count,:,:,0] 
            im = ax[i, j].imshow(img,  vmin=min, vmax=max, cmap = cmap)
            count=count +1
    fig.subplots_adjust(right=0.8)
    fig.suptitle(title)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im,  cax=cbar_ax)
    plt.show()

    
