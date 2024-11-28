import multiprocessing
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
from typing import Dict

from numpy.random import normal

from jax.numpy import exp, where, log, array
from numpy import random as numpy_random

from lmfit import Minimizer, Parameters
from lmfit import __version__ as lmfit_version
from inspect import signature
from jax import jacfwd
from jax import __version__ as jax_version
from numpy import __version__ as numpy_version

from nef_pipelines.tools.fit.fit_lib import _get_noise_from_duplicated_values
from streamfitter import __version__ as streamfitter_version

from classprop import classprop
import math

from error_propogation import ErrorPropogation


@dataclass
class PointAndValue:
    point: float
    value: float
    spectrum_name: str


class RunningStats(object):
    # after http://www.johndcook.com/blog/standard_deviation/ and B. P. Welford
    def __init__(self):
        self.m_n = 0
        self.m_oldM = 0.0
        self.m_newM = 0.0
        self.m_oldS = 0.0
        self.m_newS = 0.0

    def add(self, x):
        self.m_n += 1

        # See Knuth TAOCP vol 2, 3rd edition, page 232
        if self.m_n == 1:
            self.m_oldM = self.m_newM = x
            self.m_oldS = 0.0
        else:
            self.m_newM = self.m_oldM + (x - self.m_oldM) / self.m_n
            self.m_newS = self.m_oldS + (x - self.m_oldM) * (x - self.m_newM)

        # set up for next iteration
        self.m_oldM = self.m_newM
        self.m_oldS = self.m_newS

    def num_values(self):
        return self.m_n

    def mean(self):
        if self.m_n > 0:
            result = self.m_newM
        else:
            result = 0.0

        return result

    def variance(self):
        if self.m_n > 1:
            result = self.m_newS / self.m_n
        else:
            result = 0.0
        return result

    def stdev(self):
        return math.sqrt(self.variance())

    def sterr(self):
        if self.m_n != 0:
            result = self.stdev() / math.sqrt(self.m_n)
        else:
            result = 0.0
        return result


def _calculate_monte_carlo_error(fitter, id_xy_data, fits, error_method, noise_level, num_cycles, validate_mc=False):
    if validate_mc:
        replicate_averages = RunningStats()

    mc_fitted_params = {}
    mc_value_stats = {}
    mc_fitted_param_values = {}
    for row_count, (id, fit) in enumerate(fits.items()):
        # jax objects are not hashable
        xs = id_xy_data[id][0]
        xs_as_floats = [float(elem) for elem in xs]
        value_stats = {(id, x): RunningStats() for x in xs_as_floats}

        mc_value_stats[id] = value_stats
        mc_fitted_param_list = []
        mc_fitted_param_values[id] = mc_fitted_param_list

        fitted_params = [value.value for value in fit.params.values()]

        back_calculated = fitter.function(*fitted_params, id_xy_data[id][0])

        mc_keys_and_values = {}
        for i in range(num_cycles):
            fit_key = id, i
            mc_data = back_calculated + normal(0, noise_level, len(xs_as_floats))

            if validate_mc:
                replicate_values = {}
                for point, data in zip(xs_as_floats, mc_data):
                    replicate_values.setdefault(point, []).append(float(data))
                for replicate in replicate_values.values():
                    for combination in combinations(replicate, 2):
                        replicate_averages.add(combination[0] - combination[1])

            mc_keys_and_values[fit_key] = xs, mc_data

        fits = _fit_series(mc_keys_and_values, fitter)

        averagers = {name: RunningStats() for name in fitter.params}
        mc_calculations = 0
        for fit_key, fit in fits.items():
            if fit.success:
                mc_calculations += 1

                mc_fitted_param_list.append(fit.params)

                for name, value in fit.params.items():
                    averagers[name].add(value.value)

                mc_fitted_params_for_backcalc = [value.value for value in fit.params.values()]
                mc_back_calculated = fitter.function(*mc_fitted_params_for_backcalc, xs)
                for back_calculated, averager in zip(mc_back_calculated, value_stats.values()):
                    averager.add(back_calculated)

        errors = {f'{name}_mc_error': averager.sterr() for name, averager in averagers.items()}
        mc_fitted_params[id] = {
            **errors,
            '%mc_failures': (num_cycles - mc_calculations) / num_cycles * 100,
        }

    if validate_mc:
        print(
            f'mc mean: {replicate_averages.mean()} mc stdev: {replicate_averages.stdev()}  [from {replicate_averages.num_values()} values]'
        )

    return mc_fitted_params, mc_value_stats, mc_fitted_param_values


def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1),
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            strings.append('%s %s%s' % (period_value, period_name, has_s))

    return ', '.join(strings)


class NoNEFPipeslinesError(Exception):
    def __init__(self):
        super().__init__('nef_pipelines was not imported, stream fitter depends on NEF-Pipelines, did you install it?')


def fitter(
    id_xy_data,
    error_method: ErrorPropogation,
    cycles: int,
    noise_level,
    seed: int,
) -> Dict:
    _import_nef_pipelines_or_raise()

    id_xy_data = {id: (array(xs), array(ys)) for id, (xs, ys) in id_xy_data.items()}

    numpy_random.seed(seed)

    results = []

    replicate_noise_level, stderr, num_replicates = _get_noise_from_duplicated_values(id_xy_data)

    noise_source = 'cli' if noise_level else 'replicates'
    noise_level = noise_level or replicate_noise_level

    fitter = Relaxation2PointFitter()

    # TODO: this could be a context manager
    start_time = time.time()
    fits = _fit_series(id_xy_data, fitter)

    monte_carlo_errors, monte_carlo_value_stats, monte_carlo_param_values = _calculate_monte_carlo_error(
        fitter, id_xy_data, fits, error_method, noise_level, cycles
    )

    end_time = time.time()

    time_delta = timedelta(seconds=end_time - start_time)

    versions_string = (
        f'stream_fitter [{streamfitter_version}], lmfit [{lmfit_version}], jax[{jax_version}], numpy[{numpy_version}'
    )

    results = {
        'fits': fits,
        'monte_carlo_errors': monte_carlo_errors,
        'monte_carlo_value_stats': monte_carlo_value_stats,
        'monte_carlo_param_values': monte_carlo_param_values,
        'versions': versions_string,
        'calculation_time': time_delta,
        'number of cpus': multiprocessing.cpu_count(),
        'random seed': seed,
        'noise_level': noise_level,
        'error in noise estimate': stderr,
        'source of noise estimate': noise_source,
        'number of replicates': num_replicates,
    }

    return results


def _import_nef_pipelines_or_raise():
    try:
        import nef_pipelines  # noqa: F401
    except ImportError:
        raise NoNEFPipeslinesError()


def _get_series_variables_array(id_xy_data):
    return array([xy_data[0] for xy_data in id_xy_data.values()])


def _fit_series(ids_and_values, fitter):
    func = FunctionWrapper(fitter.function)
    jacobian = JacobianWrapper(fitter.function)

    fits = {}
    for id, (xs, ys) in ids_and_values.items():
        params = Parameters()
        estimated_parameters_dict = fitter.estimate(xs, ys)
        for key, value in estimated_parameters_dict.items():
            params.add(key, value=value)

        minimizer = Minimizer(func, params, fcn_args=(xs,), fcn_kws={'data': ys})
        out = minimizer.leastsq(Dfun=jacobian, col_deriv=1)

        fits[id] = out

    return fits


class FunctionWrapper:
    def __init__(self, func):
        self._func = func

    def __call__(self, pars, xs, data=None):
        pars = [pars[par] for par in pars]
        model = self._func(*pars, xs)
        if data is None:
            return model
        return model - data


class JacobianWrapper:
    def __init__(self, func):
        jac_args = list(range(len(signature(func).parameters) - 1))
        self._jac = jacfwd(func, jac_args)

    def __call__(self, pars, x, data=None):
        pars = [pars[par].value for par in pars]

        result = array(self._jac(*pars, x))
        return result


class Relaxation2PointFitter:
    @staticmethod
    def estimate(xs, ys):
        if xs[0] == 0.0:
            amplitude = ys[0]
        else:
            dy = ys[1] - ys[0]
            dx = xs[1] - xs[0]

            dy_dx = dy / dx

            amplitude = ys[0] - (xs[0] * dy_dx)

        delta_amplitude = ys[0] - ys[-1]
        delta_amplitude_2 = delta_amplitude / 2

        closest = min(ys, key=lambda x: abs(x - delta_amplitude_2))
        # index = [i for i, x in enumerate(ys) if x == closest][0]
        index = where(ys == closest)[0][0]

        x = xs[index]
        y = ys[index]

        time_constant = -log(y / amplitude) / x

        result = {'amplitude': amplitude, 'time_constant': time_constant.real}

        return result

    @staticmethod
    def function(amplitude, time_constant, xs):
        return amplitude * exp(-time_constant * xs)

    @classprop
    def params(cls):
        return list(signature(cls.function).parameters.keys())[:-1]
