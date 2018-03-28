"""A trust region solver."""

from flik.trustregion.tools import NotConvergedError, check_input
from flik.trustregion.cauchy import Cauchy
from flik.trustregion.dogleg import Dogleg
from flik.trustregion.subspace import Subspace
import numpy as np


def trust_region_solve(x_0, func, grad, hess, subproblem_str, reduction_factor_threshold,
                       convergence_threshold, initial_trust_region_radius=1.0,
                       maximum_trust_region_radius=1.5, maximum_number_of_iterations=128):
    r"""Minimize a scalar function with the trust region method.

    Parameters
    ----------
    x_0: np.array((N,))
        An initial guess for the solution.
    func: callable
        The scalar function that has to be minimized.
    grad: callable
        The gradient of the function that has to be minimized.
    hess: callable
        The Hessian of the function that has to be minimized, or a symmetric approximation to it.
    subproblem_str: string
        A keyword corresponding to the subproblem that should be used to solve the trust region
        sub-problem.
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


    Returns
    -------
    If converged, returns the found minimizer of func. (np.array((N,)))


    Raises
    ------
    ConvergenceError
        if the algorithm hasn't converged within the specified maximum number of iterations
    NotImplementedError
        for subproblem_str that correspond to subproblem solvers that haven't been implemented
    ValueError
        if a wrong subproblem_str was given

    """
    # Input checking
    check_input(var=x_0, func=func, grad=grad, hess=hess,
                maximum_trust_region_radius=maximum_trust_region_radius,
                initial_trust_region_radius=initial_trust_region_radius,
                reduction_factor_threshold=reduction_factor_threshold,
                maximum_number_of_iterations=maximum_number_of_iterations)

    # Do the actual algorithm
    x_k = x_0
    trust_region_radius = initial_trust_region_radius
    for k in range(1, maximum_number_of_iterations+1):  # end is not included

        # Instantiate the sub-problem solver
        if subproblem_str == 'Cauchy':
            subproblem_solver = Cauchy(x_k, func, grad, hess, trust_region_radius)
        elif subproblem_str == 'Dogleg':
            subproblem_solver = Dogleg(x_k, func, grad, hess, trust_region_radius)
        elif subproblem_str == '2D-subspace':
            subproblem_solver = Subspace(x_k, func, grad, hess, trust_region_radius)
        elif subproblem_str == 'Iterative':
            raise NotImplementedError
        else:
            raise ValueError(
                "The subproblem string did not activate a current implementation of the "
                "subproblem solvers. Did you mistype?")

        # Solve the subproblem to obtain a new step
        step = subproblem_solver.solver()

        # Calculate the reduction factor
        f_k = func(x_k)
        g_k = grad(x_k)
        hessian_k = hess(x_k)

        m0 = f_k  # m(0)
        mp_k = f_k + np.inner(g_k, step) + 0.5 * np.inner(step, np.dot(hessian_k, step))  # m(step)

        reduction_factor = (f_k - func(x_k + step)) / (m0 - mp_k)

        # Update the trust region
        if reduction_factor < 0.25:
            trust_region_radius = 0.25 * trust_region_radius
        else:
            if (reduction_factor > 0.75) and (np.linalg.norm(step) == trust_region_radius):
                trust_region_radius = min(2 * trust_region_radius, maximum_trust_region_radius)
            # The 'else' in the algorithm is redundant in code

        # Accept (or don't) the new step
        if reduction_factor > reduction_factor_threshold:
            x_k = x_k + step
        # The 'else' in the algorithm is redundant in code

        # Check for convergence
        if np.linalg.norm(grad(x_k)) < convergence_threshold:
            return x_k

    # Getting out of the loop means we have reached the maximum number of iterations
    raise NotConvergedError
