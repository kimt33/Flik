import numpy as np


class Model:
    def __init__(self, func):
        self.func = func
        # or whatever else structure for storing data
        self.cache = {}

    def update(self):
        # update cache
        pass


class LinearModel(Model):
    def __init__(self, func, grad):
        super().__init__(func)
        self.grad = grad


class QuadraticModel(Model):
    def __init__(self, func, grad, hess=None):
        super().__init__(func)
        self.grad = grad
        self.hess = hess


class HessianUpdateModel(QuadraticModel):
    def __init__(self, func, grad, hess, initial_hessian):
        super().__init__(func, grad, hess)
        self._hessian = initial_hessian

    def hessian_dot(self, vec):
        return self._hessian.dot(vec)

    def hessian_update(self, newx, newf):
        # update self.hessian
        pass


class InverseHessianUpdateModel(QuadraticModel):
    def __init__(self, func, grad, hess, initial_inv_hessian):
        super().__init__(func, grad, hess)
        self._inv_hessian = initial_inv_hessian

    def inv_hessian_dot(self, vec):
        return self._inv_hessian.dot(vec)

    def inv_hessian_update(self, newx, newf):
        # update self.inv_hessian
        pass


class ExactHessianModel(QuadraticModel):
    def __init__(self, func, grad, hess, initial_hessian):
        super().__init__(func, grad, hess)

    def hessian_dot(self, vec):
        return self.hess(vec).dot(vec)

    def inv_hessian_dot(self, vec):
        hessian = self.hess(vec)
        return np.linalg.vec(hessian).dot(vec)
