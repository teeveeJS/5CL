import numpy as np
from scipy.stats import linregress

def dy_eq(x, y, dx, dy):
    slope, _, _, _, _ = linregress(x, y)
    return np.sqrt(dy*dy + (slope * dx) * (slope * dx))

def weighted_linear_regression(x, y, dx, dy):
    # x, y, dx, and dy should be np.arrays of the data
    
    # equivalent (y) error
    deq = dy_eq(x, y, dx, dy)

    # weights
    w = deq**(-2)

    # best-fit slope
    m = (np.sum(w) * np.sum(w * x * y) - np.sum(w * x) * np.sum(w * y)) / (np.sum(w) * np.sum(w * x * x) - (np.sum(w * x))**2)

    # best-fit intercept
    b = (np.sum(w * y) - m * np.sum(w * x)) / np.sum(w)

    # uncertainty in the best-fit slope
    dm = np.sqrt(np.sum(w) / (np.sum(w) * np.sum(w * x * x) - (np.sum(w * x))**2))

    # uncertainty in the best-fit intercept
    db = np.sqrt(np.sum(w * x * x) / (np.sum(w) * np.sum(w * x * x) - (np.sum(w * x))**2))

    # coefficient of determination, r^2
    y_pred = m*x + b
    __r2 = np.sum(w * (y - y_pred)**2) / (np.sum(w * y * y) - (np.sum(w * y))**2)
    r2 = 1 - __r2

    # chi-squared (Q)
    chi2 = np.sum(((y - y_pred) / dy)**2)

    return m, dm, b, db, r2, chi2
