"""Test for flik.trustregion.trustregion."""

import numpy as np
from nose.tools import assert_raises
from flik.trustregion.trustregion import trust_region_solve


def test_trustregion():
    """Test the trustregion algorithm."""
    def func(x):
        """Return the value of the function at x."""
        return np.dot(x, x)

    def grad(x):
        """Return the value of the gradient of the function at x."""
        return 2 * x

    def hess(x):
        """Return the value of the hessian of the function at x. Only for this particular case."""
        return np.array([[2, 0], [0, 2]])

    x = np.array([1.0, 0.5])
    solution = np.array([0, 0])

    # Test input for trustregion
    assert_raises(NotImplementedError,
                  trust_region_solve, x, func, grad, hess, 'Iterative', 0.2, 10**(-5))
    assert_raises(ValueError,
                  trust_region_solve, x, func, grad, hess, 'invalid', 0.2, 10**(-5))

    # Test actual convergence
    assert np.allclose(trust_region_solve(x, func, grad, hess, 'Cauchy', 0.2, 10**(-5)), solution)
    assert np.allclose(trust_region_solve(x, func, grad, hess, 'Dogleg', 0.2, 10**(-5)), solution)
    assert np.allclose(trust_region_solve(x, func, grad, hess, '2D-subspace', 0.2, 10**(-5)),
                       solution)
