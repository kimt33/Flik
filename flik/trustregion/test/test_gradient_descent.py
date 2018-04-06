"""Tests for the Gradient-descent method."""

import numpy as np
from flik.trustregion.gradient_descent import GradientDescent


def fun(x):
    """Test function."""
    return x**3.0


def grad(x):
    """Gradient of the test function."""
    return 3.0 * x**2.0


def hess(x):
    """Hessian of the test function modified to be positive."""
    return np.array([[6.0*x[0], 0.], [0., -6.0*x[1]]])


def test_dogleg():
    """Test Dogleg method."""
    point = np.array([1.5, -0.4])
    radius = 0.7
    # Initialize class
    gradient_descent = GradientDescent(point, fun, grad, hess, radius)
    # Test when radius crosses steepest descent step
    result = gradient_descent.solver()
    assert np.allclose(result, [-0.698237, -0.0496524], atol=1.0e-6)
    # Test when it takes the steepest descent step
    gradient_descent.radius = 0.8
    result = gradient_descent.solver()
    assert np.allclose(result, [-0.752777, -0.0535308], atol=1.0e-6)
