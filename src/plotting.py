from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from truck_script import *

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')

def line_plot(ax, df, x_name, y_name, x_label, y_label, title):
    ax.plot(df[x_name].to_numpy(), df[y_name].to_numpy())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def scatter_plot(ax, df, x_name, y_name, x_label, y_label, title):
    ax.scatter(df[x_name].to_numpy(), df[y_name].to_numpy())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


if __name__=='__main__':

    df = main(df)
    df['total_minutes'].hist(bins = 50)
    plt.show()

    grouped_by_seconds = df.groupby('total_minutes').agg({'weight_of_orders':'mean'}).reset_index()
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,2)
    line_plot(ax, grouped_by_seconds, 'total_minutes', 'weight_of_orders', 'Seconds', 'Weight (lbs)', 'Stop duration by weight')
    scatter_plot(ax1, grouped_by_seconds, 'total_minutes', 'weight_of_orders', 'Seconds', 'Weight (lbs)', 'Scatter plot')
    plt.show()
