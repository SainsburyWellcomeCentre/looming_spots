import numpy as np
import scipy.signal
from sklearn import linear_model


def lerner_deisseroth_preprocess(
    photodetector_raw_data, reference_channel_211hz, reference_channel_531hz
):
    """process data according to https://www.ncbi.nlm.nih.gov/pubmed/26232229 , supplement 11"""
    demodulated_211, demodulated_531 = demodulate(
        photodetector_raw_data,
        reference_channel_211hz,
        reference_channel_531hz,
    )

    signal = apply_butterworth_lowpass_filter(demodulated_211, 2, order=2)
    background = apply_butterworth_lowpass_filter(demodulated_531, 2, order=2)

    regression_params = np.polyfit(background, signal, 1)
    bg_fit = regression_params[0] * background + regression_params[1]
    delta_f = (signal - bg_fit) / bg_fit
    return signal, background, bg_fit, delta_f


def am_demodulate(
    signal,
    reference,
    modulation_frequency,
    sample_rate=10000,
    low_cut=15,
    order=5,
):
    normalised_reference = reference - reference.mean()
    samples_per_period = sample_rate / modulation_frequency
    samples_per_quarter_period = round(samples_per_period / 4)

    shift_90_degrees = np.roll(
        normalised_reference, samples_per_quarter_period
    )

    in_phase = signal * normalised_reference
    in_phase_filtered = apply_butterworth_lowpass_filter(
        in_phase, low_cut_off=low_cut, fs=sample_rate, order=order
    )

    quadrature = signal * shift_90_degrees
    quadrature_filtered = apply_butterworth_lowpass_filter(
        quadrature, low_cut_off=low_cut, fs=sample_rate, order=order
    )

    return quadrature_filtered, in_phase_filtered


def _demodulate_quadrature(quadrature, in_phase):
    return (quadrature ** 2 + in_phase ** 2) ** 0.5


def apply_butterworth_lowpass_filter(
    demod_signal, low_cut_off=15, fs=10000, order=5
):
    w = low_cut_off / (fs / 2)  # Normalize the frequency
    b, a = scipy.signal.butter(order, w, "low")
    output = scipy.signal.filtfilt(b, a, demod_signal)
    return output


def demodulate(raw, ref_211, ref_531):

    q211, i211 = am_demodulate(raw, ref_211, 211, sample_rate=10000)
    q531, i531 = am_demodulate(raw, ref_531, 531, sample_rate=10000)
    demodulated_211 = _demodulate_quadrature(q211, i211)
    demodulated_531 = _demodulate_quadrature(q531, i531)

    return demodulated_211, demodulated_531


def robust_fit(trace):
    y = trace
    x = np.arange(len(y)).reshape(-1, 1)
    line_x = np.arange(x.min(), x.max())[:, np.newaxis]
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    line_y = ransac.predict(line_x)

    return line_y


def get_delta_f_using_robust_fit(signal, background):
    signal = signal[:-1] - robust_fit(signal)
    background = background[:-1] - robust_fit(background)

    bg_fit = background[:-1] + (robust_fit(signal) - robust_fit(background))

    delta_f = (signal[:-1] - bg_fit) / bg_fit
    return delta_f
