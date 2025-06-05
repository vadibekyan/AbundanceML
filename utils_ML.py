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


from copy import deepcopy

class FeatureRemovingAdding:
    def __init__(self, verbose=True, tolerance=1e-5):
        self.verbose = verbose
        self.tolerance = tolerance
        self.history = []

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def feature_to_remove(self, all_features, X, y, model_test_cv, maximize=True):
        best_score = model_test_cv(X[all_features], y)
        self._log(f'All features: {len(all_features)}')
        self._log(f'List of input features: {all_features}')
        self._log(f'With all features: score = {best_score:.5f}')

        best_feature_to_remove = None

        for i, feature in enumerate(all_features):
            test_feature = all_features.copy()
            test_feature.remove(feature)

            new_score = model_test_cv(X[test_feature], y)
            self._log(f'{i+1}/{len(all_features)}: Removing {feature} -> score = {new_score:.5f}')

            if ((maximize and new_score > best_score + self.tolerance) or
                (not maximize and new_score < best_score - self.tolerance)):
                self._log(f'New best score: {new_score:.5f} (was {best_score:.5f})')
                best_score = new_score
                best_feature_to_remove = feature

        return best_feature_to_remove

    def iterative_feature_removal(self, all_features, X, y, model_test_cv, min_features=2, maximize=True):
        all_features = all_features.copy()
        removed_features = []

        while len(all_features) > min_features:
            best_feature_to_remove = self.feature_to_remove(all_features.copy(), X, y, model_test_cv, maximize)
            if best_feature_to_remove is None:
                break

            removed_features.append(best_feature_to_remove)
            all_features.remove(best_feature_to_remove)
            self._log(f"Removing feature: {best_feature_to_remove}")
            self.history.append((len(all_features), model_test_cv(X[all_features], y)))

        self._log(f"Optimal set of features: {all_features}")
        self._log(f"All removed features: {removed_features}")
        return all_features

    def feature_to_add(self, all_features, removed_features, X, y, model_test_cv, maximize=True):
        best_score = model_test_cv(X[all_features], y)
        self._log(f'Current features: {len(all_features)}')
        self._log(f'With current features: score = {best_score:.5f}')

        best_feature_to_add = None

        for i, feature in enumerate(removed_features):
            test_feature = all_features.copy()
            test_feature.append(feature)

            new_score = model_test_cv(X[test_feature], y)
            self._log(f'{i+1}/{len(removed_features)}: Adding {feature} -> score = {new_score:.5f}')

            if ((maximize and new_score > best_score + self.tolerance) or
                (not maximize and new_score < best_score - self.tolerance)):
                self._log(f'New best score: {new_score:.5f} (was {best_score:.5f})')
                best_score = new_score
                best_feature_to_add = feature

        return best_feature_to_add

    def iterative_feature_adding(self, all_features, removed_features, X, y, model_test_cv, maximize=True):
        all_features = all_features.copy()
        removed_features = removed_features.copy()
        total_features = len(all_features) + len(removed_features)

        while len(all_features) < total_features:
            best_feature_to_add = self.feature_to_add(all_features.copy(), removed_features.copy(), X, y, model_test_cv, maximize)
            if best_feature_to_add is None:
                break

            all_features.append(best_feature_to_add)
            removed_features.remove(best_feature_to_add)
            self._log(f"Adding feature: {best_feature_to_add}")
            self.history.append((len(all_features), model_test_cv(X[all_features], y)))

        self._log(f"Optimal set of features: {all_features}")
        self._log(f"All remaining removed features: {removed_features}")
        return all_features
