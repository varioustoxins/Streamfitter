from jax.numpy import hstack, hsplit, array, zeros
from lmfit import Parameters

from streamfitter.ExponentialDecay2Parameter import ExponentialDecay2ParameterFitter
from streamfitter.fitter_protocol import FitterProtocol


class SharedRateExponentialDecay2ParameterFitter(FitterProtocol):
    def __init__(self):
        self._count = 2
        self._base_function = ExponentialDecay2ParameterFitter()

    def estimate(self, xs, ys):
        approximate_amplitudes = []
        approximate_time_constants = []

        xs_list = hsplit(xs, self._count)
        ys_list = hsplit(ys, self._count)

        for xs, ys in zip(xs_list, ys_list):
            values = self._base_function.estimate(xs, ys)
            approximate_time_constants.append(values['time_constant'])
            approximate_amplitudes.append(values['amplitude'])

        time_constant = (approximate_time_constants[0] + approximate_time_constants[1]) / 2.0
        return {
            'amplitude_1': approximate_amplitudes[0],
            'amplitude_2': approximate_amplitudes[1],
            'time_constant': time_constant,
        }

    def function(self, amplitude_1, amplitude_2, time_constant, xs):
        xs_1, xs_2 = hsplit(xs, self._count)

        ys_1 = self._base_function.function(amplitude_1, time_constant, xs_1)
        ys_2 = self._base_function.function(amplitude_2, time_constant, xs_2)

        return hstack((ys_1, ys_2))

    def get_wrapped_function(self):
        return MultiFunctionWrapper(self._base_function, 2)

    def get_wrapped_jacobian(self):
        return MultiFunctionJacobianWrapper(self._base_function, 2)


class MultiFunctionWrapper:
    def __init__(self, func, count):
        self._func = func
        self._count = count

    def __call__(self, pars, xs, data=None):
        pars = [pars[par] for par in pars]
        amplitude_1, amplitude_2, time_constant = pars

        parameters_1 = Parameters()
        parameters_1.add(amplitude_1.name, amplitude_1.value)
        parameters_1.add(time_constant.name, time_constant.value)

        parameters_2 = Parameters()
        parameters_2.add(amplitude_2.name, amplitude_2.value)
        parameters_2.add(time_constant.name, time_constant.value)

        split_pars = parameters_1, parameters_2
        split_xs = hsplit(xs, self._count)
        split_data = hsplit(data, self._count) if data is not None else [None] * self._count

        values = []
        function_wrapper = self._func.get_wrapped_function()
        for one_pars, one_xs, one_data in zip(split_pars, split_xs, split_data):
            values.append(function_wrapper(one_pars, one_xs, one_data))

        return hstack(values)


class MultiFunctionJacobianWrapper:
    def __init__(self, func, count):
        self._func = func
        self._count = count

    def __call__(self, pars, xs, data=None):
        pars = [pars[par] for par in pars]
        amplitude_1, amplitude_2, time_constant = pars

        parameters_1 = Parameters()
        parameters_1.add(amplitude_1.name, amplitude_1.value)
        parameters_1.add(time_constant.name, time_constant.value)

        parameters_2 = Parameters()
        parameters_2.add(amplitude_2.name, amplitude_2.value)
        parameters_2.add(time_constant.name, time_constant.value)

        split_pars = parameters_1, parameters_2
        split_xs = hsplit(xs, self._count)
        split_data = hsplit(data, self._count) if data is not None else [None] * self._count

        jacs = []
        function_wrapper = self._func.get_wrapped_jacobian()
        for one_pars, one_xs, one_data in zip(split_pars, split_xs, split_data):
            jacs.append(function_wrapper(one_pars, one_xs, one_data))

        len_x = len(xs) // 2

        result = array(
            [hstack([jacs[0][0], zeros(len_x)]), hstack([zeros(len_x), jacs[1][0]]), hstack([jacs[0][1], jacs[1][1]])]
        )

        return result
