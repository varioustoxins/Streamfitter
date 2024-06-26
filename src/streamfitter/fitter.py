import multiprocessing
import time
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum, auto
from itertools import combinations
from typing import List, Dict

from nef_pipelines.lib.shift_lib import IntensityMeasurementType
from nef_pipelines.lib.util import exit_error
from numpy.random import normal


from statistics import stdev

from pynmrstar import Entry, Saveframe
from jax.numpy import exp, where, log, array
from numpy import random as numpy_random

from lmfit import Minimizer, Parameters
from lmfit import __version__ as lmfit_version
from inspect import signature
from jax import jacfwd
from jax import __version__ as jax_version
from numpy import __version__ as numpy_version
from streamfitter import __version__ as streamfitter_version

from classprop import classprop
import math


class ErrorPropogation(StrEnum):
    PROPOGATION = auto()
    JACKNIFE = auto()
    BOOTSTRAP = auto()


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


def _calculate_monte_carlo_error(fitter, xs, fits, error_method, noise_level, num_cycles, validate_mc=False):
    xs_as_floats = [float(x) for x in xs]
    if validate_mc:
        replicate_xs = [value for value, count in Counter(xs_as_floats).items() if count > 1]
        replicate_averages = RunningStats()

    mc_fitted_params = {}
    mc_value_stats = {}
    mc_fitted_param_values = {}
    for row_count, (atom_key, fit) in enumerate(fits.items()):
        value_stats = {(i, value): RunningStats() for i, value in enumerate(xs_as_floats)}
        mc_value_stats[atom_key] = value_stats
        mc_fitted_param_list = []
        mc_fitted_param_values[atom_key] = mc_fitted_param_list

        fitted_params = [value.value for value in fit.params.values()]

        back_calculated = fitter.function(*fitted_params, xs)

        mc_keys_and_values = {}
        for i in range(num_cycles):
            fit_key = atom_key, i
            mc_data = back_calculated + normal(0, noise_level, len(xs))

            if validate_mc:
                replicate_values = {}
                for point, data in zip(xs_as_floats, mc_data):
                    if point in replicate_xs:
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
        mc_fitted_params[atom_key] = {
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
    entry: Entry,
    series_frames: List[Saveframe],
    error_method: ErrorPropogation,
    cycles: int,
    noise_level,
    intensity_measurement_type: IntensityMeasurementType,
    seed: int,
) -> Dict:
    _import_nef_pipelines_or_raise()

    from nef_pipelines.lib.peak_lib import frame_to_peaks

    numpy_random.seed(seed)

    results = []
    for series_frame in series_frames:
        series_experiment_loop = get_loop(series_frame, 'nef_series_experiment')

        _exit_if_no_series_lists_selected(series_frame, series_experiment_loop)

        spectra_by_times_and_indices = _get_spectra_by_series_variable(entry, series_experiment_loop)

        _exit_if_spectra_are_missing(spectra_by_times_and_indices, series_frame.name)

        peaks_by_times_and_indices = {
            key: frame_to_peaks(spectrum_frame) for key, spectrum_frame in spectra_by_times_and_indices.items()
        }

        atoms_and_values = _get_atoms_and_values(peaks_by_times_and_indices, intensity_measurement_type)

        replicate_noise_level, stderr, num_replicates = _get_noise_from_duplicated_values(
            peaks_by_times_and_indices, intensity_measurement_type
        )

        noise_source = 'cli' if noise_level else 'replicates'
        noise_level = noise_level or replicate_noise_level

        fitter = Relaxation2PointFitter()

        # TODO: this could be a context manager
        start_time = time.time()
        fits = _fit_series(atoms_and_values, fitter)

        xs = _get_series_variables_array(series_experiment_loop)

        monte_carlo_errors, monte_carlo_value_stats, monte_carlo_param_values = _calculate_monte_carlo_error(
            fitter, xs, fits, error_method, noise_level, cycles
        )
        end_time = time.time()

        time_delta = timedelta(seconds=end_time - start_time)

        versions_string = f'stream_fitter [{streamfitter_version}], lmfit [{lmfit_version}], jax[{jax_version}], numpy[{numpy_version}'

        results.append(
            {
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
        )

    return results


def _get_noise_from_duplicated_values(peaks_by_times_and_indices, data_type: IntensityMeasurementType) -> float:
    peak_list_keys_by_times = {}
    for key, peak_list in peaks_by_times_and_indices.items():
        _, value, _ = key
        peak_list_keys_by_times.setdefault(value, []).append(key)

    duplicated_peak_list_keys = [
        peak_list_keys for peak_list_keys in peak_list_keys_by_times.values() if len(peak_list_keys) > 1
    ]

    differences = []
    for duplicated_peak_list_set in duplicated_peak_list_keys:
        for combination in combinations(duplicated_peak_list_set, 2):
            peak_pairs = {}
            for key in combination:
                for peak in peaks_by_times_and_indices[key]:
                    if data_type == IntensityMeasurementType.HEIGHT:
                        value = peak.height
                    elif data_type == IntensityMeasurementType.VOLUME:
                        value = peak.volume
                    key = tuple(sorted([shift.atom for shift in peak.shifts]))
                    peak_pairs.setdefault(key, []).append(value)

            for value_pair in peak_pairs.values():
                if len(value_pair) == 2:
                    differences.append(value_pair[0] - value_pair[1])

    replicates_stdev = stdev(differences)
    replicates_stderr = replicates_stdev / len(differences) ** 0.5

    return replicates_stdev, replicates_stderr / replicates_stdev, len(differences)


def _import_nef_pipelines_or_raise():
    try:
        import nef_pipelines  # noqa: F401
    except ImportError:
        raise NoNEFPipeslinesError()


def _get_series_variables_array(series_experiment_loop):
    from nef_pipelines.lib.nef_lib import loop_row_namespace_iter

    return array([row.series_variable for row in loop_row_namespace_iter(series_experiment_loop, convert=True)])


def _fit_series(atoms_and_values, fitter):
    func = FunctionWrapper(fitter.function)
    jacobian = JacobianWrapper(fitter.function)

    fits = {}
    for atom_key, (xs, ys) in atoms_and_values.items():
        xs = array(xs)
        ys = array(ys)
        params = Parameters()
        estimated_parameters_dict = fitter.estimate(xs, ys)
        for key, value in estimated_parameters_dict.items():
            params.add(key, value=value)

        minimizer = Minimizer(func, params, fcn_args=(xs,), fcn_kws={'data': ys})
        out = minimizer.leastsq(Dfun=jacobian, col_deriv=1)

        fits[atom_key] = out

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


def _get_atoms_and_values(peaks_by_times_and_indices, data_type: IntensityMeasurementType) -> Dict:
    atoms_to_values = {}
    for (index, x_value, spectrum_name), peaks in peaks_by_times_and_indices.items():
        for peak in peaks:
            atoms = tuple(sorted([shift.atom for shift in peak.shifts]))
            x_and_y = atoms_to_values.setdefault(atoms, [[], []])

            y_value = peak.height if data_type == IntensityMeasurementType.HEIGHT else peak.volume
            x_and_y[0].append(x_value)
            x_and_y[1].append(y_value)

    return atoms_to_values


def _get_spectra_by_series_variable(entry, series_experiment_loop):
    from nef_pipelines.lib.nef_lib import loop_row_namespace_iter

    spectra_by_times = {}
    for i, row in enumerate(loop_row_namespace_iter(series_experiment_loop, convert=True), start=1):
        spectrum_frame_name = row.nmr_spectrum_id
        series_variable = row.series_variable

        spectrum_frame = entry.get_saveframe_by_name(spectrum_frame_name)

        key = i, series_variable, spectrum_frame_name
        spectra_by_times[key] = spectrum_frame

    return spectra_by_times


def _exit_if_no_series_lists_selected(frame, series_experiment_loop):
    if not series_experiment_loop:
        msg = f'no nef series experiment loop found in frame {frame.name}'
        exit_error(msg)


# TODO: move to lib
def get_loop(frame, loop_name):
    try:
        series_experiment_loop = frame.get_loop(loop_name)
    except KeyError:
        series_experiment_loop = None
    return series_experiment_loop


def _exit_if_spectra_are_missing(spectra_by_times_and_indices, series_name):
    for (
        _,
        _,
        spectrum_frame_name,
    ), spectrum_frame in spectra_by_times_and_indices.items():
        if not spectrum_frame:
            msg = f'no spectrum frame found for series {series_name} for spectrum {spectrum_frame_name}'
            exit_error(msg)
