"""Lightcurve processing.
"""
import numpy as np
import tensorflow as tf


class RaggedTensorCurve(tf.RaggedTensor):
    """Ragged light curve.

    Attributes:
        ragged_curve (tf.RaggedTensor): ragged curve.
        sampling_rate (float): sampling rate.
    """
    def extract_sequential_curve(self, lower_size, truncate=False):
        rc_extracted = tf.ragged.boolean_mask(
            self, self.row_lengths() >= lower_size)

        if truncate:
            return rc_extracted[:, :lower_size, :]
        else:
            return rc_extracted

    @classmethod
    def load_curve(cls, curve, sampling_rate, dtype=tf.float32):

        if isinstance(sampling_rate, (int, float)) is False:
            raise ValueError("sampling_rate must be `float` or `int`.")

        curve = tf.convert_to_tensor(curve, dtype=dtype)

        cls._sampling_rate = sampling_rate

        times = curve[:, 0] - curve[0, 0]
        time_diffs = np.round(np.diff(times), 6)  # round at micro second.
        splits = tf.concat(
            [
             [0],
             tf.reshape(tf.where(time_diffs != sampling_rate), [-1]) + 1,
            ],
            axis=-1)
        return cls.from_row_starts(curve, splits)

    def filter_gti(self, gtis):
        """Filter Good Time Interval (GTI).

        Args:
            * gtis (int or float): 2-d tensor of good time interval.

        Return:
            * A RaggedTensorCurve
        """
        gtis = tf.convert_to_tensor(gtis, dtype=self.dtype)

        is_greater_equal = tf.math.greater_equal(
            self[..., 0][:, tf.newaxis],
            gtis[..., 0][:, tf.newaxis])
        is_less = tf.math.less(
            self[..., 0][:, tf.newaxis],
            gtis[..., 1][:, tf.newaxis])
        is_in_gti = tf.cast(
            tf.math.reduce_max(
                tf.cast(tf.math.logical_and(is_greater_equal, is_less),
                        tf.int32),
                axis=-2),
            tf.bool
        )

        return self.load_curve(
            tf.ragged.boolean_mask(self, is_in_gti).flat_values,
            self.sampling_rate)

    @property
    def sampling_rate(self):
        return self._sampling_rate
