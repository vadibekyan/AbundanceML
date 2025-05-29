#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error



def score_dataset(X, y, model, scoring='MAE', cv=5):
    """
    Calculate the mean and std of MAE using cross-validation.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        model: The machine learning model.
        scoring (str): Scoring method.
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: (mean_MAE, std_MAE)
    """
    # Encode categoricals
    X = X.copy()
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()

    # Cross-validation scores
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1  # Use all CPUs
    )
    mae_scores = -scores
    return mae_scores.mean(), mae_scores.std()

def modified_learning_curve(X_train, y_train, X_test, y_test, model, N, CV=5, display=False, figure=True):
    """
    Calculate and plot the modified learning curve for a given model.

    Parameters:
        X_train (array-like): The input features of the training dataset.
        y_train (array-like): The target variable of the training dataset.
        X_test (array-like): The input features of the test dataset.
        y_test (array-like): The target variable of the test dataset.
        model: The machine learning model used for prediction.
        N (int): The number of data points to be used in the learning curve.
        CV (int): The number of cross-validation folds (default: 5).
        display (bool): Whether to display intermediate training results (default: False).
        figure (bool): Whether to plot the learning curve (default: True).

    Returns:
        DataFrame: The learning curve results.

    """

    # Generate an array of training sizes
    train_sizes = np.linspace(0.05, 0.99, N)

    # Create an empty DataFrame to store the learning curve results
    results = pd.DataFrame(columns=['train_size', 'mae_train_mean', 'mae_train_std', 'mae_test_mean', 'mae_test_std'],
                           index=range(0, N))

    # Iterate over the training sizes
    for i, train_size in enumerate(train_sizes):
        mae_train = []
        mae_test = []

        # Repeat the cross-validation process for each training size
        for j in range(CV):
            # Split the training set into a subset based on the current training size
            X_train_tmp, _, y_train_tmp, _ = train_test_split(X_train, y_train, test_size=(1 - train_size))

            # Fit the model on the subset of training data
            model.fit(X_train_tmp, y_train_tmp)

            # Make predictions on the test set
            y_pred_test = model.predict(X_test)
            y_pred_train_tmp = model.predict(X_train_tmp)

            # Evaluate the model using mean absolute error
            mae_train.append(mean_absolute_error(y_train_tmp, y_pred_train_tmp))
            mae_test.append(mean_absolute_error(y_test, y_pred_test))

        # Store the learning curve results in the DataFrame
        results['train_size'][i] = len(X_train_tmp)
        results['mae_train_mean'][i] = np.mean(mae_train)
        results['mae_train_std'][i] = np.std(mae_train)
        results['mae_test_mean'][i] = np.mean(mae_test)
        results['mae_test_std'][i] = np.std(mae_test)

        # Print intermediate results if display is True
        if display:
            print(f'For a training size of {len(X_train_tmp)}, MAE(train) = {np.mean(mae_train):.3f} and MAE(test) = {np.mean(mae_test):.3f}')

    # Convert object columns to numeric type in the DataFrame
    for column in results.columns:
        if results[column].dtype == 'object':
            results[column] = pd.to_numeric(results[column], errors='coerce')

    # Plot the learning curve if figure is True
    if figure:
        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")

        plt.plot(results['train_size'], results['mae_train_mean'], 'o-', color="r", label="Training Score")
        plt.fill_between(results['train_size'], results['mae_train_mean'] + results['mae_train_std'],
                         results['mae_train_mean'] - results['mae_train_std'],
                         alpha=0.15, color='r')

        plt.plot(results['train_size'], results['mae_test_mean'], 'o-', color="g", label="Cross-Validation Score")
        plt.fill_between(results['train_size'], results['mae_test_mean'] + results['mae_test_std'],
                         results['mae_test_mean'] - results['mae_test_std'],
                         alpha=0.15, color='g')

        plt.legend(loc="best")
        plt.show()

    return results
