import unittest
from utils.runs import compute_bias_amplification
import numpy as np
import pandas as pd


class TestBACelebA(unittest.TestCase):
    def test_ba_perfect_predictions(self):
        targets = np.zeros([100, 40])
        targets[:50, 0] = 1
        targets[:25, 1:] = 1
        targets[50:70, 1:] = 1
        predictions = np.zeros([100, 40])
        predictions[:50, 0] = 1
        predictions[:25, 1:] = 1
        predictions[50:70, 1:] = 1
        larger_fracs_df = pd.DataFrame(np.ones([100, 40])*0.6)
        smaller_fracs_df = pd.DataFrame(np.ones([100, 40])*0.4)

        bas = compute_bias_amplification(targets, predictions, 0, 1, 'celeba', larger_fracs_df, smaller_fracs_df)
        self.assertEqual(bas, 0.0)

    def test_ba_imperfect_predictions(self):
        targets = np.zeros([100, 40])
        targets[50:, 0] = 1
        targets[:25, 1] = 1
        targets[50:75, 1] = 1
        predictions = np.zeros([100, 40])
        predictions[50:, 0] = 1
        predictions[:25, 1] = 1
        predictions[50:80, 1] = 1
        larger_fracs_df = pd.DataFrame(np.ones([100, 40])*0.6)
        smaller_fracs_df = pd.DataFrame(np.ones([100, 40])*0.4)

        bas = compute_bias_amplification(targets, predictions, 0, 1, 'celeba', larger_fracs_df, smaller_fracs_df)
        reverse_bas = compute_bias_amplification(targets, predictions, 0, 1, 'celeba', smaller_fracs_df, larger_fracs_df)
        self.assertEqual(bas, 30/55 - 0.5)
        self.assertEqual(reverse_bas, 25/55 - 0.5)


if __name__ == '__main__':
    unittest.main()
