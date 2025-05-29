#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns


from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def y_distribution_plot(y):
    from sklearn.neighbors import KernelDensity

    # Perform kernel density estimation (KDE) on the 'parameter' column
    kde = KernelDensity(bandwidth=0.2).fit(y.values.reshape(-1, 1))
    # Calculate the density of each value in the 'parameter' column
    densities = np.exp(kde.score_samples(y.values.reshape(-1, 1)))

    # Create the figure and gridspec for customizing subplot sizes
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0)

    # Define the upper and lower subplots using gridspec
    ax_boxplot = fig.add_subplot(gs[0])
    ax_histogram = fig.add_subplot(gs[1])

    # Plot the boxplot in the top subplot
    boxplot_dict = ax_boxplot.boxplot(y, vert=False, showfliers=True, widths=0.5)
    # Hide x-axis labels of the top subplot
    ax_boxplot.xaxis.set_ticklabels([])

    # Adjust the line width of the boxplot elements
    for element in ['boxes', 'whiskers', 'caps']:
        plt.setp(boxplot_dict[element], linewidth=2)  # Adjust the linewidth here


    # Extract the outliers from the boxplot dictionary
    outliers = boxplot_dict['fliers'][0].get_data()

    # Plot the histogram in the bottom subplot
    ax_histogram.hist(y, bins=30, density = True, alpha = 0.5)
    ax_histogram.scatter(y, densities, color = 'black')

    # Add labels and adjust layout
    plt.xlabel('[Fe/H]')
    plt.tight_layout()


def corr_heatmap_plot(df):
    #Correlation heatmap
    corr = df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr))
    fig = plt.subplots(figsize=(10,5))
    sns.heatmap(corr, annot=True,fmt='.2f',mask=mask,cmap='coolwarm_r', cbar=None, linewidth=1)
    plt.suptitle('Correlation Heatmaap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def ols_fit(x, y):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    #print(results.summary())
    coeff1_ols, coeff2_ols = results.params[0], results.params[1]
    return coeff1_ols, coeff2_ols


def make_mi_scores(X, y, n_features=None, plot=True):
    """
    Compute and visualize Mutual Information (MI) scores for feature selection.

    Parameters:
    - X: DataFrame
        The feature matrix.
    - y: Series
        The target variable.
    - n_features: int or None, optional (default=None)
        Number of top features to display. If None, it defaults to the total number of features.
    - plot: bool, optional (default=True)
        Whether to plot the MI scores.

    Returns:
    - mi_scores: Series
        Mutual Information scores for each feature.

    The function computes MI scores using mutual_info_regression and optionally plots a bar chart
    of the top features based on their MI scores.

    Note: The function assumes that discrete features have been encoded as integers.
    """
    X = X.copy()

    # Label encoding for categoricals
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # Set n_features to the length of mi_scores if not provided
    if n_features is None:
        n_features = len(mi_scores)

    print(f'First {n_features} of {len(mi_scores)} features:\n{mi_scores[:n_features]}')

    if plot:
        if len(mi_scores) > n_features:
            mi_scores_to_plot = mi_scores[:n_features]
        else:
            mi_scores_to_plot = mi_scores

        fig, axs = plt.subplots(1, 1)
        scores_sorted = mi_scores_to_plot.sort_values(ascending=True)
        width = np.arange(len(scores_sorted))
        ticks = list(scores_sorted.index)
        plt.barh(width, scores_sorted)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
        # Set up figure
        fig.set(figwidth=8, dpi=100)

    return mi_scores



def preprocess_data(X_train, X_test=None, standardize=True, factorize=True):
    """
    Preprocess training and testing data by label-encoding or factorizing categorical columns
    and optionally standardizing numeric columns.

    Parameters:
    - X_train: DataFrame
        The training data to be preprocessed.
    - X_test: DataFrame, optional (default=None)
        The testing data to be preprocessed. If None, only X_train will be preprocessed.
    - standardize: bool, optional (default=True)
        Whether to standardize numeric columns using StandardScaler.
    - factorize: bool, optional (default=True)
        Whether to factorize categorical columns using pandas factorize. If False, label encoding will be skipped.

    Returns:
    - X_train_processed: DataFrame
        The preprocessed training data with label-encoded or factorized categorical columns
        and optionally standardized numeric columns.
    - X_test_processed: DataFrame or None
        The preprocessed testing data with the same preprocessing steps as X_train. None if X_test is not provided.
    - scaler: StandardScaler or None
        The fitted scaler. If standardize is False, returns None.

    The function creates a copy of the input DataFrames, encodes categorical columns, and optionally standardizes numeric columns.
    The original DataFrames are not modified, and the preprocessed DataFrames and fitted scaler are returned.
    """
    X_train = X_train.copy()  # Create a copy to avoid modifying the original data

    if X_test is not None:
        X_test = X_test.copy()  # Create a copy of the testing data if provided

    # Encoding for categoricals using factorize or label encoding
    mapping_dict = {}

    for colname in X_train.select_dtypes(["object", "category"]):
        if factorize:
            X_train[colname], mapping = X_train[colname].factorize()
            mapping_dict[colname] = mapping
        else:
            original_dtype = X_train[colname].dtype
            X_train[colname] = X_train[colname].astype('category').cat.codes.astype(original_dtype)

    # Apply the same mapping to the testing data
    if X_test is not None:
        for colname, mapping in mapping_dict.items():
            X_test[colname] = X_test[colname].map(mapping).fillna(-1)  # Handle unknown categories

    # Identify numeric columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    scaler = None

    # Standardize numeric columns (if specified)
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X_train[num_cols])
        X_train[num_cols] = scaler.transform(X_train[num_cols])

        if X_test is not None:
            X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test


def apply_pca(X_train, X_test=None, num_components=None, plot=True):
    # Create principal components using training data
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)

    # Convert training data to dataframe
    component_names = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=component_names)

    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X_train.columns,
    )
    if plot:
        # Create figure
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)

        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))

        # Set up figure
        fig.set(figwidth=8, dpi=100)

    if X_test is not None:
        # Apply the same transformation to the test data
        X_test_pca = pca.transform(X_test)
        X_test_pca_df = pd.DataFrame(X_test_pca, columns=component_names)
        return pca, X_train_pca_df, X_test_pca_df, loadings

    return pca, X_train_pca_df, loadings