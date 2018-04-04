"""Test for flik.trustregion.subspace."""
import numpy as np
from flik.trustregion.subspace import Subspace
from flik.trustregion.cauchy import Cauchy


def test_subspace():
    """Test the subspace solver."""
    # Set parameters
    point = np.array([1., 2.])  # Leading to positive definite hessian
    zero_point = np.array([0., 1/6])  # Leading to hessian with 0 eigenvalue
    neg_point = np.array([1/6, -1/6])  # Leading tot negative definite hessian

    def func(point):
        """Objective function."""
        return point**3

    def grad(point):
        """Gradient of the objective function."""
        return 3*point**2

    def hess(point):
        """Hessian of the objective function."""
        return np.array([[6*point[0], 0.], [0., 6*point[1]]])
    radius = 1.
    sub_sol = Subspace(point, func, grad, hess, radius)
    zero_sol = Subspace(zero_point, func, grad, hess, radius)
    neg_sol = Subspace(neg_point, func, grad, hess, radius)
    cauchy = Cauchy(zero_point, sub_sol.func, sub_sol.grad,
                    sub_sol.hess, radius)

    assert np.array_equal(sub_sol.point, np.array([1., 2.]))
    assert np.array_equal(sub_sol.func(sub_sol.point), np.array([1., 8.]))
    assert np.array_equal(sub_sol.grad(sub_sol.point), np.array([3., 12.]))
    assert np.array_equal(sub_sol.hess(sub_sol.point),
                          np.array([[6., 0.], [0., 12.]]))
    assert np.array_equal(sub_sol.gradient, np.array([3., 12.]))
    assert np.array_equal(sub_sol.hessian, np.array([[6., 0.], [0., 12.]]))
    assert sub_sol.min_hess == 6.0
    assert np.allclose(sub_sol._aux_grad(sub_sol.hessian),
                       np.array([np.sqrt(153.), 13.5]))
    assert np.allclose(sub_sol._aux_hess(sub_sol.hessian),
                       np.array([[1782., np.sqrt(153.)],
                                 [np.sqrt(153.), 13.5]]))
    assert np.allclose(sub_sol._aux_norm(sub_sol.hessian),
                       np.array([[153., 13.5], [13.5, 1.11803399]]))
    assert np.array_equal(sub_sol.free_step(hess(point), grad(point)),
                          np.array([-0.5, -1.]))
    assert sub_sol.solution_norm(hess(point), grad(point)) == 891.
    assert sub_sol.aux_polynomial(sub_sol.hessian, sub_sol.hessian,
                                  sub_sol.gradient,
                                  sub_sol.radius) == [250776.,
                                                      501552.,
                                                      250774.]
    assert np.allclose(sub_sol._solution(np.array([0.1, 0.2]), sub_sol.hessian),
                       np.array([0.4, 1.4]))
    # Hessian with 0 eigenvalue
    assert np.allclose(zero_sol.solver(), cauchy.solver())
    # Positive hessian, inside trust region
    assert np.allclose(sub_sol.solver(), np.array([-0.5, -1.]))
    # Positive hessian, small trust region
    small_region = Subspace(point, func, grad, hess, radius=0.4)
    assert np.allclose(small_region.solver(), np.array([0.42390953, 1.47969983]))
    # Negative hessian, inside trust region
    assert np.allclose(neg_sol.solver(), np.array([-1/30, -1/6]))
    # Negative hessian, small trust region
    small_region = Subspace(neg_point, func, grad, hess, radius=0.2)
    assert np.allclose(small_region.solver(), np.array([-0.00568462,
                                                        -0.00480769]))
