import numpy as np
from scipy.stats import linregress

def chi_squared(x, y, dx, dy, weighted=False):
    m, b, _, _, _ = linregress(x, y)
    y_pred = m*x + b
    delta_y = 0
    w = 1
    if weighted:
        from weighted_linear_regression import dy_eq
        delta_y = dy_eq(x, y, dx, dy)
        w = delta_y**(-2)
    else:
        delta_y = dy
    return np.sum((w*(y - y_pred) / delta_y)**2)

def main():
    # some tests
    x1 = np.array([0.48, 0.469, 0.416, 0.219, 0.213])
    y1 = np.array([0.225, 0.235, 0.291, 0.482, 0.493])
    dx1 = np.array([0.002, 0.002, 0.002, 0.002, 0.002])
    dx2 = np.array([0.002, 0.002, 0.002, 0.002, 0.002])

    print(chi_squared(x1, y1, dx1, dx2))
    print(chi_squared(x1, y1, dx1, dx2, True))

    x2 = np.array([-2.286, -2.190, -2.095, -0.524, -0.524])
    y2 = np.array([-2.133, -1.996, -1.430, -0.454, -0.432])
    dy2 = np.array([0.021, 0.019, 0.012, 0.0046, 0.0044])
    print(chi_squared(x2, y2, None, dy2))


if __name__ == "__main__":
    main()
