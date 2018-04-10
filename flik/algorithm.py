def algorithm(x, model, subproblem_solver, trustregion, niter):
    for i in range(niter):
        # solve model (using given algorithm)
        step = subproblem_solver(x, model, trustregion)
        # update trust region and model
        trustregion.update(x, step, model)
        model.update(x, step)
        # update x
        if trustregion.check_step(step):
            x += step
    return x
