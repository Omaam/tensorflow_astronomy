"""
"""
import numpy as np
import tensorflow as tf

import tensorflow_astronomy as tfa


class TestRaggedTensorCurve(tf.test.TestCase):

    def test_shape(self):
        time_ranges = [
            [0, 100],
            [105, 153],
            [160, 200],
        ]
        times = [np.arange(r[0], r[1], 1) for r in time_ranges]
        times = np.concatenate(times, axis=0)
        count_rates = np.random.poisson(30, times.shape[0])
        lc = np.stack([times, count_rates], axis=1)

        rc = tfa.RaggedTensorCurve.load_curve(lc, 1.0)

        self.assertEqual((3, None, 2), rc.shape)


if __name__ == "__main__":
    tf.test.main()
