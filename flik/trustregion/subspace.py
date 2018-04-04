"""Subspace solver for the sub-problem."""
import numpy as np
from flik.trustregion.sub_problem import SubProblem
from flik.trustregion.cauchy import Cauchy


class Subspace(SubProblem):
    """Subspace solver.

    Attributes
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
    gradient: np.ndarray((N,))
        Gradient at the current point.
    hessian: np.ndarray((N,N,))
        Hessian at the current point.
    min_hess: float
        Minimum eigenvalue of the hessian.

    Methods
    -------
    __init__(self, point, function, grad, hessian, radius)
        Initialize the data corresponding to the given point.
    _aux_grad(self, ahess)
        Calculate auxiliary gradient.
    _aux_hess(self, ahess)
        Calculate auxiliary hessian.
    _aux_norm(self, ahess)
        Calculate auxiliary matrix.
    free_step(ahess, agrad)
        Unconstrained step (static method).
    solution_norm(ahess, grad)
        Weighted norm of a vector (static method).
    aux_polynomial(self, mhess, nhess, agrad, radius)
        Form auxiliary polynomial (static method).
    _solution(self, ssol, ahess)
        Solution in original basis
    solver(self)
        Sub-problem subspace solver.

    """

    def __init__(self, point, func, grad, hess, radius):
        """Initialize the data corresponding to the given point.

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
        self.gradient = self.grad(self.point)
        self.hessian = self.hess(self.point)
        self.min_hess = min(np.linalg.eig(self.hessian)[0])

    def _aux_grad(self, ahess):
        """Calculate auxiliary gradient.

        A modified gradient in the basis spanning the subspace.

        Parameters
        ----------
        ahess: np.ndarray((N,N,))
            Matrix (approximated hessian) used to modify the gradient.

        Returns
        -------
        np.ndarray((N,)):
            The modified gradient.

        """
        return np.array([np.linalg.norm(self.gradient),
                         self.gradient.dot(np.linalg.inv(ahess).dot(self.gradient))])

    def _aux_hess(self, ahess):
        """Calculate auxiliary hessian.

        A modified hessian in the basis spanning the subspace.

        Parameters
        ----------
        ahess: np.ndarray((N,N,))
            (Approximated) hessian.

        Returns
        -------
        np.ndarray((N,N,))
            The modified hessian.

        """
        return np.array([[self.gradient.dot(ahess.dot(self.gradient)),
                          np.linalg.norm(self.gradient)],
                         [np.linalg.norm(self.gradient),
                          self.gradient.dot(np.linalg.inv(ahess).dot(self.gradient))]])

    def _aux_norm(self, ahess):
        """Calculate auxiliary matrix.

        A matrix used to calculate the norm of a vector
        given in the basis spanning the subspace.

        Parameters
        ----------
        ahess: np.ndarray((N,N,))
            (Approximated) hessian.

        Returns
        -------
        np.ndarray((N,N,))
            The matrix used to calculate the norm.


        """
        return np.array([[(np.linalg.norm(self.gradient))**2,
                          self.gradient.dot(np.linalg.inv(ahess).dot(self.gradient))],
                         [self.gradient.dot(np.linalg.inv(ahess).dot(self.gradient)),
                          np.linalg.norm(np.linalg.inv(ahess).dot(self.gradient))]])

    @staticmethod
    def free_step(ahess, agrad):
        """Unconstrained step.

        The solution of the Newton problem.

        Parameters
        ----------
        ahess: np.ndarray((N,N,))
            (Approximated) hessian.
        agrad: np.ndarray((N,))
            (Approximated) gradient.

        Returns
        -------
        np.ndarray((N,))
            Unconstrained optimization step.

        """
        return -np.linalg.inv(ahess).dot(agrad)

    @staticmethod
    def solution_norm(ahess, agrad):
        """Weighted norm of a vector.

        Parameters
        ----------
        ahess: np.ndarray((N,N,))
            (Approximated) hessian.
        agrad: np.ndarray((N,))
            (Approximated) gradient.

        Returns
        -------
        float
            Weighted norm.

        """
        return 0.5*agrad.dot(ahess.dot(agrad))

    @staticmethod
    def aux_polynomial(mhess, nhess, agrad, radius):
        """Form auxiliary polynomial.

        Parameters
        ----------
        mhess: np.ndarray((N,N,))
            Modified hessian.
        nhess: np.ndarray((N,N,))
            Matrix used to calculate the norm.
        agrad: np.ndarray((N,))
            (Approximated) gradient.
        radius: float
            Trust region radius

        Returns
        -------
        list
            A list with the coefficients of the 2nd order polynomial.

        """
        qterm = (nhess.dot(agrad)).dot(nhess.dot(nhess.dot(agrad)))
        lterm = ((mhess.dot(agrad)).dot(nhess.dot(nhess.dot(agrad))) +
                 (nhess.dot(agrad)).dot(nhess.dot(mhess.dot(agrad))))
        cterm = ((mhess.dot(agrad)).dot(nhess.dot(mhess.dot(agrad))) -
                 2*radius**2)
        return [qterm, lterm, cterm]

    def _solution(self, ssol, ahess):
        """Solution in original basis.

        Parameters
        ----------
        ssol: np.ndarray((N,))
            Solution in the basis spanning the subspace.
        ahess: np.ndarray((N,N,))
            (Approximated) hessian.

        Returns
        -------
        np.ndarray((N,))
            Solution in the original basis.

        """
        return ssol[0]*self.gradient+ssol[1]*np.linalg.inv(ahess).dot(self.gradient)

    def solver(self):
        """Subspace sub-problem solver.

        Returns
        -------
        step: np.ndarray((N,))
            Correction added to the current point.

        """
        if self.min_hess == 0:
            cauchy = Cauchy(self.point, self.func, self.grad,
                            self.hess, self.radius)
            return cauchy.solver()
        elif self.min_hess > 0:
            new_hess = self.hessian
        else:
            alpha = -1.5*self.min_hess
            new_hess = self.hessian + alpha*np.identity(self.hessian.ndim)

        mhess = self._aux_hess(new_hess)
        nhess = self._aux_norm(new_hess)
        new_grad = self._aux_grad(new_hess)
        u_step = Subspace.free_step(mhess, new_grad)
        sol_norm = Subspace.solution_norm(nhess, u_step)
        if sol_norm < self.radius**2:
            return self._solution(u_step, new_hess)
        else:
            roots = np.roots(self.aux_polynomial(mhess, nhess,
                                                 new_grad, self.radius))
            solutions = [-np.linalg.inv(mhess+root*nhess).dot(new_grad)
                         for root in roots]
            values = [solution.dot(new_grad)+0.5*solution.dot(mhess.dot(solution))
                      for solution in solutions]
            return self._solution(solutions[int(np.argmin(values))], new_hess)
