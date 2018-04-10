"""Problem of finding the minima in the given trust region."""
from flik.model import HessianUpdateModel, InverseHessianUpdateModel, ExactHessianModel


class HessianDotSubproblem:
    def __init__(self, model):
        if not isinstance(model, (HessianUpdateModel, ExactHessianModel)):
            raise ValueError
        self.model = model

    def solve(self, x, trustregion):
        raise NotImplementedError


class InvHessianDotSubproblem:
    def __init__(self, model):
        if not isinstance(model, (InverseHessianUpdateModel, ExactHessianModel)):
            raise ValueError
        self.model = model

    def solve(self, x, trustregion):
        raise NotImplementedError


class Dogleg(InvHessianDotSubproblem):
    pass


class Subspace(InvHessianDotSubproblem):
    pass


class CauchyPoint(InvHessianDotSubproblem):
    pass


class Steinhaug(HessianDotSubproblem):
    pass


class Iterative(HessianDotSubproblem):
    pass


# OR ALTERNATIVELY
def solve_dogleg(x, model, trustregion):
    if not isinstance(model, (InverseHessianUpdateModel, ExactHessianModel)):
        raise ValueError


def solve_subspace(x, model, trustregion):
    if not isinstance(model, (InverseHessianUpdateModel, ExactHessianModel)):
        raise ValueError


def solve_cauchy_point(x, model, trustregion):
    if not isinstance(model, (InverseHessianUpdateModel, ExactHessianModel)):
        raise ValueError


def solve_steinhaug(x, model, trustregion):
    if not isinstance(model, (HessianUpdateModel, ExactHessianModel)):
        raise ValueError


def solve_iterative(x, model, trustregion):
    if not isinstance(model, (HessianUpdateModel, ExactHessianModel)):
        raise ValueError
