import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.
    """
    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, true_coef


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers.
    """
    rng = np.random.RandomState(random_state)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:n_outliers] = 10 + 0.75 * rng.normal(size=(n_outliers, 1))
    y_out[:n_outliers] = -15 + 20 * rng.normal(size=n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot dataset with highlighted outliers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[n_outliers:],
        y[n_outliers:],
        label="Normal observations"
    )

    ax.scatter(
        X[:n_outliers],
        y[:n_outliers],
        label="Artificial outliers"
    )

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_huber_regression(X, y):
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    errors = {}

    for name, coef in coef_dict.items():
        errors[name] = abs(coef - true_coef)

    return errors


def best_robust_model(errors):
    robust_errors = {
        "huber_regression": errors["huber_regression"],
        "ransac_regression": errors["ransac_regression"],
        "theilsen_regression": errors["theilsen_regression"]
    }

    return min(robust_errors, key=robust_errors.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = np.sum(outlier_mask)

    added_outliers_detected = np.sum(
        outlier_mask[:n_outliers]
    )

    return (
        int(total_outliers_detected),
        int(added_outliers_detected)
    )


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot regression lines from all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X, y, alpha=0.5, label="Data")

    X_line = np.linspace(
        X.min(),
        X.max(),
        500
    ).reshape(-1, 1)

    models = {
        "Linear Regression":
            LinearRegression(),

        "Huber Regression":
            HuberRegressor(),

        "RANSAC Regression":
            RANSACRegressor(
                random_state=random_state
            ),

        "Theil-Sen Regression":
            TheilSenRegressor(
                random_state=random_state
            )
    }

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X_line)

        ax.plot(
            X_line,
            y_pred,
            label=name
        )

    ax.set_title("Regression Model Fits")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Plot RANSAC inliers and outliers.
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[inlier_mask],
        y[inlier_mask],
        label="Inliers"
    )

    ax.scatter(
        X[outlier_mask],
        y[outlier_mask],
        label="Outliers"
    )

    ax.set_title("RANSAC Inliers vs Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig