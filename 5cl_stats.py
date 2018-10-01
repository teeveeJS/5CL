import numpy as np

"""
a quick and dirty implementation of the core statistical functions
used in Physics 5CL
"""

"""
Functions in single variable
"""

def mean(x):
    return np.sum(x) / len(x)

def mean_dev(x, xi):
    # deviation from the mean
    return xi - mean(x)

def var(x):
    # variance
    return mean(x*x) - mean(x)**2

def stddev_pop(x):
    # population standard deviation
    return np.sqrt(var(x))

def stddev_s(x):
    # sample standard deviation
    return np.sqrt(np.sum(np.apply_along_axis(lambda xi: mean_dev(x, xi), 0, x)**2) / (len(x) - 1))

def std_err(x):
    # standard error
    return sttdev_s(x) / np.sqrt(len(x))


"""
Functions in two variables
"""

def cov(x, y):
    # covariance
    return mean(x * y) - mean(x) * mean(y)

def coeff_corr(x, y, sample=True):
    # coefficient of linear correlation, r
    # set sample=False to compute r for a population standard deviation
    if sample:
        return cov(x, y) / (stddev_s(x) * stddev_s(y))
    else:
        return cov(x, y) / (stddev_pop(x) * stddev_pop(y))


"""
Simple least-squares linear regression
(could apply to non-linear functions as well)
"""

# TODO: sum of the squares of the residuals

def lin_slope(x, y):
    # best-fit slope for the linear hypothesis
    return (mean(x * y) - mean(x) * mean(y)) / (mean(x * x) - mean(x)**2)

def lin_intercept(x, y):
    # best-fit y-intercept for the linear hypothesis
    return (mean(x * x) * mean(y) - mean(x) * mean(x * y)) / (mean(x * x) - mean(x)**2)

def lin_dy(x, y):
    # uncertainties in y based on the linear fit
    y_pred = lin_slope(x, y) * y + lin_intercept(x, y)
    return np.sqrt(np.sum((y - y_pred)**2) / (len(x) - 2))

def lin_dm(x, y):
    # uncertainty in the best-fit slope
    return lin_dy(x, y) / np.sqrt(len(x) * var(x))

def lin_db(x, y):
    # uncertainty in the best-fit y-intercept
    return lin_dy(x, y) * np.sqrt(mean(x * x) / (len(x) * var(x)))

def y_pred(x, y):
    return lin_slope(x, y) * x + lin_intercept(x, y)

def lin_regress(x, y):
    return lin_slope(x, y), lin_dm(x, y), lin_intercept(x, y), lin_db(x, y)

"""
Direct proportionality hypothesis (b = 0)
"""

def lin_dir_m(x, y):
    # slope of the direct proportionality linear hypothesis
    return mean(x * y) / mean(x * x)

def lin_dir_dy(x, y):
    # uncertainty in y based on the d.p.l.h
    return np.sqrt(np.sum((y - lin_dir_m * x)**2) / (len(x) - 1))

def lin_dir_dm(x, y):
    # uncertainty in the slope of the d.p.l.h.
    return lin_dir_dy(x, y) / np.sqrt(len(x) * mean(x * x))


"""
Weighed Linear Regression
* Currently implemented in another file
"""


"""
Coefficient of Determination and Chi-Squared Tests
"""

def r2(x, y):
    # coefficient of determination, r^2
    return 1 - np.sum((y - y_pred(x, y))**2) / (np.sum(y * y) - (np.sum(y))**2)

# TODO weighted r2

def r2_adj(x, y, p=1):
    return r2(x, y) - p * (1 - r2(x, y)) / (len(x) - p - 1)

def chi_squared(x, y, dy):
    return np.sum(((y - y_pred(x, y)) / dy)**2)

# TODO chi-squared without dy?

def chi_squared_red(x, y, dy, np=2):
    # np is the number of parameters in the fit; equal to 2 in most cases
    return chi_squared(x, y, dy) / (len(x) - np)
