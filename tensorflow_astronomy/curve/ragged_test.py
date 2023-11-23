"""
"""
import numpy as np
import tensorflow as tf

import tensorflow_astronomy as tfa


class TestRaggedTensorCurve(tf.test.TestCase):

    def test_shape(self):

        time_ranges = [
            [0, 2],
            [5, 9],
            [15, 17],
        ]
        times = [np.arange(r[0], r[1], 1) for r in time_ranges]
        times = np.concatenate(times, axis=0)
        count_rates = np.random.poisson(30, times.shape[0])
        lc = np.stack([times, count_rates], axis=1)
        rc = tfa.RaggedTensorCurve.load_curve(lc, 1.0)

        self.assertEqual((3, None, 2), rc.shape)

    def test_gti(self):
        times = tf.constant(
            [1., 2., 5., 6., 7.,
             8., 15., 16., 17.],
        )
        count_rates = 10 * np.ones(times.shape[-1])
        lc = np.stack([times, count_rates], axis=1)
        rc = tfa.RaggedTensorCurve.load_curve(lc, 1.0)

        gtis = tf.constant([[1, 2], [3, 7], [8, 16], [17, 20]],
                           dtype=tf.float32)

        rc_gti_filtered = rc.filter_gti(gtis)

        expected = tf.constant([
            [1, 10],
            [5, 10],
            [6, 10],
            [8, 10],
            [15, 10],
            [17, 10],
        ])
        self.assertAllEqual(expected, rc_gti_filtered.flat_values)


if __name__ == "__main__":
    tf.test.main()
