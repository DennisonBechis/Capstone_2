from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from cleaning_script import *

df = pd.read_csv('/Users/bechis/dsi/repo/Capstone_2/data/training_dataset.csv')

def line_plot(ax, df, x_name, y_name, x_label, y_label, title = None):
    ax.plot(df[x_name].to_numpy(), df[y_name].to_numpy())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def scatter_plot(ax, df, x_name, y_name, x_label, y_label, title = None):
    ax.scatter(df[x_name].to_numpy(), df[y_name].to_numpy())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def plot_mnist_embedding(ax, X, y, title=None):
    """Plot an embedding of the mnist dataset onto a plane.

    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.

    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.

    y: numpy.array
      The labels of the datapoints.  Should be digits.

    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i]), s = 3, alpha = 0.2)

    ax.set_xticks([]),
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)

def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.

    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.

    pca: sklearn.decomposition.PCA object.
      A fit PCA object.

    n_components_to_plot: int
      The number of principal components to display in the skree plot.

    title: str
      A title for the skree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]),
                   (ind[i]+0.2, vals[i]+0.005),
                   va="bottom",
                   ha="center",
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)

if __name__=='__main__':

    df = main(df)
    df['total_minutes'].hist(bins = 62)

    plt.style.use('classic')
    fig3 = plt.figure(figsize=(6,6))
    ax3 = fig3.add_subplot(1,1,1)
    ax3.hist(df['total_minutes'].to_numpy(), bins=62)
    ax3.set_xlabel('Stop Time (Minutes)')
    ax3.set_ylabel('Count')
    plt.show()

    grouped_by_seconds = df.groupby('total_minutes').agg({'weight_of_orders':'mean'}).reset_index()
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(2,1,1)
    ax1 = fig.add_subplot(2,1,2)
    line_plot(ax, grouped_by_seconds, 'total_minutes', 'weight_of_orders', 'Minutes', 'Weight (lbs)')
    scatter_plot(ax1, grouped_by_seconds, 'total_minutes', 'weight_of_orders', 'Minutes', 'Weight (lbs)')
    plt.tight_layout()
    plt.show()
