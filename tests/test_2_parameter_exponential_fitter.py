import numpy as np
from jax.numpy import array
from pytest import approx

from streamfitter.ExponentialDecay2Parameter import ExponentialDecay2ParameterFitter
from streamfitter.error_propogation import ErrorPropogation
from streamfitter.fitter import fit

TIME_CONSTANT_1_3 = 1.3
AMPLITUDE_1_5 = 1.5

# amplitude = 1.5 time_constant = 1.3
EXPONENTIAL_A_1_5 = array(
    [
        1.5,
        0.8417159529712340,
        0.47232383032418200,
        0.2650416686348950,
        0.14872653379473800,
        0.08345699741676420,
        0.046831390741846400,
        0.026279152458161000,
        0.014746387903064900,
        0.008274846631141157,
    ]
)

XS = np.linspace(0, 4, 10)


def test_exponential_estimator():
    instance = ExponentialDecay2ParameterFitter()
    result = instance.estimate(XS, EXPONENTIAL_A_1_5)

    assert approx(AMPLITUDE_1_5) == result['amplitude']
    assert approx(TIME_CONSTANT_1_3) == result['time_constant']


def test_exponential_function():
    instance = ExponentialDecay2ParameterFitter()
    result = instance.function(AMPLITUDE_1_5, TIME_CONSTANT_1_3, XS)

    assert approx(EXPONENTIAL_A_1_5) == result


def test_fitter_with_exponential():
    key = 1

    # this forces the fitter to do some work as the guess
    # of the initial amplitude will be off as it just uses the largest
    # values in the ys
    ys = EXPONENTIAL_A_1_5[1:]
    xs = XS[1:]

    id_xy_data = {key: [xs, ys]}

    fitter = ExponentialDecay2ParameterFitter()
    result = fit(fitter, id_xy_data, ErrorPropogation.PROPOGATION, None, None, 42)

    estimates = result['estimates'][key]

    # this proves that the fitter is doing the work and not the stimator
    estimated_amplitude = estimates['amplitude']
    estimated_time_constant = estimates['time_constant']

    assert approx(AMPLITUDE_1_5) != estimated_time_constant
    assert approx(TIME_CONSTANT_1_3) != estimated_amplitude

    fits = result['fits'][key]
    amplitude = fits.params['amplitude']
    time_constant = fits.params['time_constant']

    assert approx(AMPLITUDE_1_5) == amplitude
    assert approx(TIME_CONSTANT_1_3) == time_constant
