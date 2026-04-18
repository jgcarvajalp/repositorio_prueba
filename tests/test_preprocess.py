import unittest
from src.preprocessing.preprocess import scale_features
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    def test_scale_features(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = scale_features(df.copy(), ['a', 'b'])
        self.assertAlmostEqual(result['a'].mean(), 0, places=6)
        self.assertAlmostEqual(result['b'].mean(), 0, places=6)

if __name__ == '__main__':
    unittest.main()
