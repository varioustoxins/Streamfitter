from jax.numpy import exp, where, log
from streamfitter.fitter_protocol import FitterProtocol


class ExponentialDecay2ParameterFitter(FitterProtocol):
    def estimate(self, xs, ys):
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

        result = {'amplitude': amplitude, 'time_constant': time_constant.tolist()}

        return result

    def function(self, amplitude, time_constant, xs):
        return amplitude * exp(-time_constant * xs)
