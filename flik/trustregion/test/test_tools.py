"""Test for flik.trustregion tools."""

import numpy as np
from nose.tools import assert_raises
from flik.trustregion.tools import check_input


def test_input_control():
    """Test flik.trustregion.tools.check_input."""
    # check var
    assert_raises(TypeError, check_input, var=[1])
    assert_raises(TypeError, check_input, var=np.array([[1, 2]]))
    check_input(var=0)
    check_input(var=0.0)
    check_input(var=np.array([0]))

    # check func
    assert_raises(TypeError, check_input, func='1')
    check_input(func=lambda x: x)

    # check grad
    assert_raises(TypeError, check_input, grad='1')
    check_input(grad=lambda x: x)

    # check hess
    assert_raises(TypeError, check_input, hess='1')

    def test_hess(x, y):
        return np.array([[x, 2*y**2], [2*y**2, x*y]])
    check_input(hess=test_hess)

    # check func_val
    assert_raises(TypeError, check_input, func_val=np.array([1]))
    check_input(func_val=1)
    check_input(func_val=1.0)

    # check grad_val
    assert_raises(TypeError, check_input, grad_val=[1])
    check_input(grad_val=1)
    check_input(grad_val=1.0)
    check_input(grad_val=np.array([1]))

    # check hess_val
    assert_raises(TypeError, check_input, hess_val=[1])
    assert_raises(TypeError, check_input, hess_val=np.array([1, 2.0]))
    assert_raises(ValueError, check_input, hess_val=np.array([[1, 2.0], [-2.0, 3]]))
    check_input(hess_val=1)
    check_input(hess_val=1.0)
    check_input(hess_val=np.array([[1, 2.0], [2.0, 3]]))

    # check step
    assert_raises(TypeError, check_input, step=[1])
    assert_raises(TypeError, check_input, step=np.array(1))
    check_input(step=1)
    check_input(step=1.0)
    check_input(step=np.array([1]))

    # check var and step
    assert_raises(ValueError, check_input, var=1, step=np.array([2, 3]))
    assert_raises(ValueError, check_input, var=np.array([1]), step=np.array([2, 3]))
    check_input(var=1, step=np.array([2]))

    # check var and grad_val
    assert_raises(ValueError, check_input, var=np.array([2, 3]), grad_val=1)
    assert_raises(ValueError, check_input, var=np.array([2, 3]), grad_val=np.array([1]))
    check_input(var=np.array([2]), grad_val=1)

    # check grad_val and step
    assert_raises(ValueError, check_input, grad_val=1, step=np.array([2, 3]))
    assert_raises(ValueError, check_input, grad_val=np.array([1]), step=np.array([2, 3]))
    check_input(grad_val=1, step=np.array([2]))

    # check var, grad_val and step
    assert_raises(ValueError, check_input, var=1, grad_val=1, step=np.array([2, 3]))
    check_input(var=0, grad_val=1, step=np.array([2]))

    # check hess_val and step
    assert_raises(ValueError, check_input, hess_val=np.array([[1, 2.0], [2.0, 3]]),
                  step=np.array([2.0]))
    check_input(hess_val=np.array([[1, 2.0], [2.0, 3]]), step=np.array([2.0, 0.0]))

    # check maximum_trust_region_radius
    assert_raises(TypeError, check_input, maximum_trust_region_radius=[1])
    assert_raises(ValueError, check_input, maximum_trust_region_radius=-0.5)
    check_input(maximum_trust_region_radius=3.0)

    # check initial_trust_region_radius
    assert_raises(TypeError, check_input, initial_trust_region_radius=[1])
    assert_raises(ValueError, check_input, initial_trust_region_radius=-0.5)
    assert_raises(ValueError, check_input, initial_trust_region_radius=0.03,
                  maximum_trust_region_radius=0.02)
    check_input(initial_trust_region_radius=0.001, maximum_trust_region_radius=0.02)

    # check reduction_factor_threshold
    assert_raises(TypeError, check_input, reduction_factor_threshold=[1])
    assert_raises(ValueError, check_input, reduction_factor_threshold=-0.5)
    assert_raises(ValueError, check_input, reduction_factor_threshold=0.5)
    check_input(reduction_factor_threshold=0.1)

    # check convergence_threshold
    assert_raises(TypeError, check_input, convergence_threshold=[1])
    assert_raises(ValueError, check_input, convergence_threshold=-0.5)
    check_input(convergence_threshold=0.1)

    # check maximum_number_of_iterations
    assert_raises(TypeError, check_input, maximum_number_of_iterations=[1])
    assert_raises(TypeError, check_input, maximum_number_of_iterations=2.5)
    assert_raises(ValueError, check_input, maximum_number_of_iterations=0)
    assert_raises(ValueError, check_input, maximum_number_of_iterations=-2)
    check_input(maximum_number_of_iterations=12)

    # do nothing
    check_input()
