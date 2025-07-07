import numpy as np
from jax.numpy import hstack, array
from pytest import approx

from streamfitter.SharedRateExponential2Parameter import SharedRateExponentialDecay2ParameterFitter
from streamfitter.error_propogation import ErrorPropogation
from streamfitter.fitter import fit

TIME_CONSTANT_1_3 = 1.3
AMPLITUDE_1_5 = 1.5
AMPLITUDE_2_5 = 2.5

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

# amplitude = 2.5 time_constant = 1.3
EXPONENTIAL_A_2_5 = array(
    [
        2.5,
        1.402859921618724,
        0.7872063838736371,
        0.44173611439149185,
        0.24787755632456326,
        0.13909499569460698,
        0.07805231790307726,
        0.043798587430268286,
        0.024577313171774806,
        0.013791411051901929,
    ]
)

DOUBLE_EXPONENTIAL_SHARED_RATE = hstack((EXPONENTIAL_A_1_5, EXPONENTIAL_A_2_5))

XS = np.linspace(0, 4, 10)


def test_shared_rate_exponential_estimator():
    xs = np.hstack([XS, XS])
    ys = np.hstack([EXPONENTIAL_A_1_5, EXPONENTIAL_A_2_5])

    function = SharedRateExponentialDecay2ParameterFitter()
    result = function.estimate(xs, ys)

    assert approx(AMPLITUDE_1_5) == result['amplitude_1']
    assert approx(AMPLITUDE_2_5) == result['amplitude_2']
    assert approx(TIME_CONSTANT_1_3) == result['time_constant']


def test_paired_exponential_function():
    xs_2 = hstack((XS, XS))

    function = SharedRateExponentialDecay2ParameterFitter()
    result = function.function(AMPLITUDE_1_5, AMPLITUDE_2_5, TIME_CONSTANT_1_3, xs_2)

    assert approx(DOUBLE_EXPONENTIAL_SHARED_RATE) == result


def test_fitter_with_paired_exponential():
    # this forces the fitter to do some work as the guess
    # of the initial amplitude will be off as it just uses the largest
    # values in the ys
    ys_1 = EXPONENTIAL_A_1_5[1:]
    ys_2 = EXPONENTIAL_A_2_5[1:]
    xs_1_2 = XS[1:]

    xs = np.hstack([xs_1_2, xs_1_2])
    ys = np.hstack([ys_1, ys_2])

    key = 1

    id_xy_data = {key: [xs, ys]}

    fitter = SharedRateExponentialDecay2ParameterFitter()

    result = fit(fitter, id_xy_data, ErrorPropogation.PROPOGATION, None, None, 42)

    estimates = result['estimates'][key]

    # this proves that the fitter is doing the work and not the stimator
    estimated_amplitude_1 = estimates['amplitude_1']
    estimated_amplitude_2 = estimates['amplitude_2']
    estimated_time_constant = estimates['time_constant']

    assert approx(AMPLITUDE_1_5) != estimated_amplitude_1
    assert approx(AMPLITUDE_2_5) != estimated_amplitude_2
    assert approx(TIME_CONSTANT_1_3) != estimated_time_constant

    fits = result['fits'][key]
    amplitude_1 = fits.params['amplitude_1']
    amplitude_2 = fits.params['amplitude_2']
    time_constant = fits.params['time_constant']

    assert approx(AMPLITUDE_1_5) == amplitude_1
    assert approx(AMPLITUDE_2_5) == amplitude_2
    assert approx(TIME_CONSTANT_1_3) == time_constant
