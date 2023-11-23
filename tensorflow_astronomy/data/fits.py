"""Fits handling functions
"""
import warnings

import tensorflow as tf
from astropy.io import fits


def load_curve_fits(lc_file, dtype=tf.float32):
    with fits.open(lc_file) as hdul:
        lcdata = hdul[1].data
        time = lcdata["TIME"]
        rate = lcdata["RATE"]
        error = lcdata["ERROR"]

    if time[-1].bit_length() >= 32:
        warnings.warn(
            "For compelte load time information,"
            "more than 32 bits are needed.")

    curve = tf.stack(
        [tf.convert_to_tensorflow(time, tf.float32),
         tf.convert_to_tensorflow(rate, tf.float32),
         tf.convert_to_tensorflow(error, tf.float32)],
        axis=-1)

    return curve
