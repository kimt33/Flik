"""Tools for the trust region module."""

import itertools as it
import numpy as np


def check_input(*, var=None, func=None, grad=None, hess=None, func_val=None, grad_val=None,
                hess_val=None, step=None, maximum_trust_region_radius=None,
                initial_trust_region_radius=None, reduction_factor_threshold=None,
                convergence_threshold=None, maximum_number_of_iterations=None):
    r"""Test input used in trust region module.

    Parameters
    ----------
    var: {int, float, np.array((N,)), None}
        Variables' current values.
    func: {callable, None}
        Function :math:`f(x)`.
    grad: {callable, None}
        Gradient of the function :math:`\nabla f(x)`.
    hess: {callable, None}
        Hessian of the function :math:`\nabla f(x)` or an approximation to it.
    func_val: {int, float, None}
        Value of the function evaluated at var.
    grad_val: {int, float, np.array((N,)), None}
        Gradient evaluated at var.
    hess_val: {float, np.array((N,N,)), None}
        Hessian evaluated at var.
    step: {int, float, np.ndarray((N,))}
        Direction vector for next step.
    maximum_trust_region_radius: {int, double}
        The maximum radius the trust region can obtain.
    initial_trust_region_radius: {int, double}
        The initial trust region radius, should be in [0, maximum_trust_region_radius[
    reduction_factor_threshold: {int, double}
        The threshold used for accepting a reduction factor.
    convergence_threshold: {int, double}
        The threshold used to determine convergence.
    maximum_number_of_iterations: int
        A maximum of the number of iterations the trust region solver should perform.


    Raises
    ------
    TypeError
        If var is not an int, float, or one dimensional numpy array.
        If func is not a callable.
        If grad is not a callable.
        If hess is not callable.
        If func_val is not an int or float.
        If grad_val is not an int, float, or one-dimensional numpy array.
        If hess_val is not an int, float, or two-dimensional numpy array.
        If step is not an int, float, or one dimensional numpy array.
    ValueError
        If var and direction do not have the same shape.
        If var and grad_val do not have the same shape.
        If step and grad_val do not have the same shape.
        If hess_val is incompatible with step.
        If hess_val is not a symmetric matrix.
        If maximum_trust_region_radius is smaller than zero.
        If initial_trust_region_radius is not in the interval ]0, maximum_trust_region_radius[.
        If reduction_factor_threshold is not in the interval [0, 0.25[
        If maximum_number_of_iterations is not larger than zero.

    """
    if var is None:
        pass
    elif isinstance(var, (int, float)):
        var = np.array(var, dtype=float, ndmin=1)
    elif not (isinstance(var, np.ndarray) and var.ndim == 1):
        raise TypeError("Variable vector should be a float or a 1-D numpy.ndarray")

    if func is None:
        pass
    elif not callable(func):
        raise TypeError("func must be callable")

    if grad is None:
        pass
    elif not callable(grad):
        raise TypeError("grad must be callable")

    if hess is None:
        pass
    elif not callable(hess):
        raise TypeError("hess must be callable")

    if func_val is None:
        pass
    elif not isinstance(func_val, (int, float)):
        raise TypeError('func_val must be a float.')

    if grad_val is None:
        pass
    elif isinstance(grad_val, (int, float)):
        grad_val = np.array(grad_val, dtype=float, ndmin=1)
    elif not (isinstance(grad_val, np.ndarray) and grad_val.ndim == 1):
        raise TypeError('grad_val must be a one dimensional numpy array.')

    if hess_val is None:
        pass
    elif isinstance(hess_val, (int, float)):
        hess_val = np.array(hess_val, dtype=float, ndmin=2)
    elif not (isinstance(hess_val, np.ndarray) and hess_val.ndim == 2):
        raise TypeError('hess_val must be a two dimensional numpy array.')
    elif not np.allclose(hess_val, hess_val.T):
        raise ValueError("The current Hessian must be a symmetric matrix.")

    if step is None:
        pass
    elif isinstance(step, (int, float)):
        step = np.array(step, dtype=float, ndmin=1)
    elif not (isinstance(step, np.ndarray) and step.ndim == 1):
        raise TypeError("The direction vector should be provided as a float or a numpy array")

    name_dict = {'var': var, 'direction': step, 'grad_val': grad_val}
    for name1, name2 in it.combinations(name_dict, 2):
        array1 = name_dict[name1]
        array2 = name_dict[name2]

        if array1 is None or array2 is None:
            continue
        if array1.shape != array2.shape:
            raise ValueError(f'{name1} and {name2} must have the same shape.')

    if hess_val is not None and step is not None:
        hess_val_dim_1, hess_val_dim_2 = hess_val.shape
        step_dim = step.shape[0]  # step is already checked to have dimension (1,)
        if (hess_val_dim_1 != step_dim) or (hess_val_dim_2 != step_dim):
            raise ValueError("hess_val and step are not compatible.")

    if maximum_trust_region_radius is None:
        pass
    elif not isinstance(maximum_trust_region_radius, (int, float)):
        raise TypeError("maximum_trust_region_radius must be an int or a float.")
    elif maximum_trust_region_radius <= 0:
        raise ValueError("maximum_trust_region_radius must be larger than zero.")

    if initial_trust_region_radius is None:
        pass
    elif not isinstance(initial_trust_region_radius, (int, float)):
        raise TypeError("initial_trust_region_radius must be an int or a float.")
    elif (initial_trust_region_radius <= 0) or \
            (maximum_trust_region_radius < initial_trust_region_radius):
        raise ValueError("initial_trust_region_radius must be in the interval ]0, "
                         "maximum_trust_region_radius[")

    if reduction_factor_threshold is None:
        pass
    elif not isinstance(reduction_factor_threshold, (int, float)):
        raise TypeError("reduction_factor_threshold must be an int or a float.")
    elif (reduction_factor_threshold > 0.25) or (reduction_factor_threshold < 0):
        raise ValueError("reduction_factor_threshold must be in the interval [0, 0.25[")

    if convergence_threshold is None:
        pass
    elif not isinstance(convergence_threshold, (int, float)):
        raise TypeError("convergence_threshold must be an int or a float.")
    elif convergence_threshold < 0:
        raise ValueError("convergence_threshold must be larger than 0.")

    if maximum_number_of_iterations is None:
        pass
    elif not isinstance(maximum_number_of_iterations, int):
        raise TypeError("maximum_number_of_iterations must be an int.")
    elif maximum_number_of_iterations <= 0:
        raise ValueError("maximum_number_of_iterations must be larger than zero.")
