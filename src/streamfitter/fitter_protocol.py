from inspect import signature
from typing import Protocol
from jax import jacfwd
from jax.numpy import array


class FitterProtocol(Protocol):
    def estimate(self, xs, ys):
        ...

    def function(self, *args, xs):
        ...

    def get_wrapped_function(self):
        return FunctionWrapper(self)

    def get_wrapped_jacobian(self):
        return JacobianWrapper(self)

    def params(self):
        return list(signature(self.function).parameters.keys())[:-1]


class FunctionWrapper:
    def __init__(self, func):
        self._func = func

    def __call__(self, pars, xs, data=None):
        pars = [pars[par] for par in pars]
        model = self._func.function(*pars, xs)
        if data is None:
            return model
        return model - data


class JacobianWrapper:
    def __init__(self, func):
        jac_args = list(range(len(signature(func.function).parameters) - 1))
        self._jac = jacfwd(func.function, jac_args)

    def __call__(self, pars, x, data=None):
        pars = [pars[par].value for par in pars]

        result = array(self._jac(*pars, x))
        return result
