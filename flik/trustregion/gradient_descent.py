"""Gradient descent method for the subproblem of the trust-region method."""

import numpy as np

from flik.trustregion.sub_problem import SubProblem


__all__ = ['GradientDescent']


class GradientDescent(SubProblem):
    r"""Gradient Descent method.

    Attributes
    ----------
    point: np.ndarray (1-dimensional)
        Initial point of the optimization.
    func: callable
        Objective function.
    grad: callable
        Gradient of the objective function.
    hess: callable
        Hessian of the objective function.
    current_gradient: np.ndarray (1-dimensional)
        Gradient of the objective function at the current point.
    current_hessian: np.ndarray (2-dimensional)
        Hessian of the objective function at the current point.
    radius: float
        The trust region radius.

    Methods
    -------
    __init__(self, point, function, grad, hessian, radius, *params)
        Initialize the data corresponding to the given point.
    solver(self)
        Subproblem Gradient descent solver.

    """

    def __init__(self, point, func, grad, hess, radius):
        """Initialize the method.

        Parameters
        ----------
        point: np.ndarray((N,))
            Initial point of the optimization.
        func: callable
            Objective function.
        grad: callable
            Gradient of the objective function.
        hess: callable
            Hessian of the objective function.
        radius: float
            The trust region radius.

        """
        super().__init__(point, func, grad, hess, radius)
        self.current_gradient = self.grad(self.point)
        self.current_hessian = self.hess(self.point)

    def solver(self):
        """Find new step.

        Returns
        -------
        step: np.ndarray((N,))
            Correction added to the current point.

        """
        # Compute steepest-descent direction step
        grad_squared = np.dot(self.current_gradient.T, self.current_gradient)
        hessian_term = np.dot(self.current_hessian, self.current_gradient)
        hessian_term = np.dot(self.current_gradient.T, hessian_term)
        steepest_step = - (grad_squared / hessian_term) * self.current_gradient
        if np.linalg.norm(steepest_step) >= self.radius:
            step = (self.radius/np.linalg.norm(steepest_step))*steepest_step
        else:
            step = steepest_step
        return step
