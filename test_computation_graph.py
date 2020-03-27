import unittest
import numpy as np
from computation_graph import ComputationNode, Values, DotProduct, Sum, Softmax, Relu, Flatten
from scipy import special


class ComputationGraphTests(unittest.TestCase):

    def test_values_set_values_shape_none(self):
        """
        Test Values set_values() correctly handles inputs when first element of shape is None
        """
        x = Values(shape=(None, 10))
        success = True
        try:
            x.set_values(np.ones(shape=(100, 10)))
        except AssertionError:
            success = False
        self.assertTrue(success)
        success = True
        try:
            x.set_values(np.ones(shape=(100, 12)))
        except AssertionError:
            success = False
        self.assertTrue(not success)

    def test_dotproduct_eval(self):
        x = Values(shape=(None, 1, 5))
        x.set_values(np.random.normal(0, 1, size=(10, 1, 5)))

        y = Values(shape=(5, 3))
        y.set_values(np.random.normal(0, 1, size=y.shape))

        z = DotProduct()([x, y])
        res = z.eval()
        expected = np.dot(x.values, y.values)
        self.assertTrue(np.array_equal(res, expected))

    def test_flatten_output_shape_when_parent_shape_none(self):
        x = Values(shape=(None, 10, 10, 3))  # Some 10 x 10 x 3 inputs
        x.set_values(np.ones(shape=(100, 10, 10, 3)))  # 100 inputs of shape 10 x 10 x 3

        flatten = Flatten()(x)
        self.assertEqual(flatten.shape, (None, 300))

    def test_flatten_output_shape_when_parent_shape_fixes(self):
        x = Values(shape=(10, 10, 3))  # Single input of shape 10 x 10 x 3
        x.set_values(np.ones(shape=(10, 10, 3)))

        flatten = Flatten()(x)
        self.assertEqual(flatten.shape, (300,))

    def test_flatten_eval_when_parent_shape_none(self):
        x = Values(shape=(None, 10, 10, 3))  # Some 10 x 10 x 3 inputs
        x.set_values(np.ones(shape=(200, 10, 10, 3)))  # 200 inputs

        flatten = Flatten()(x)
        res = flatten.eval()
        self.assertEqual(res.shape, (200, 300))

    def test_softmax_when_parent_shape_none_vectors(self):
        x = Values(shape=(None, 1, 5))  # Any given number of 1x5 vectors
        vals = np.array([[[1, 2, 3, 4, 5]], [[2, 3, 4, 5, 6]], [[1, 1, 1, 1, 1]]])
        x.set_values(vals)
        softmax = Softmax()(x)
        self.assertEqual(softmax.shape, x.shape)
        res = softmax.eval()
        self.assertEqual(res.shape, x.values.shape)
        for val_vector, res_vector in zip(x.values, res):
            scipy_res = special.softmax(val_vector)
            equal = np.allclose(res_vector, scipy_res, rtol=1e-6)
            self.assertTrue(equal)
            self.assertAlmostEqual(1., np.sum(res_vector))

    def test_softmax_when_parent_shape_fixed_vectors(self):
        x = Values(shape=(3, 1, 5))  # Any given number of 1x5 vectors
        vals = np.array([[[1, 2, 3, 4, 5]], [[2, 3, 4, 5, 6]], [[1, 1, 1, 1, 1]]])
        x.set_values(vals)
        softmax = Softmax()(x)
        self.assertEqual(softmax.shape, x.shape)
        res = softmax.eval()
        self.assertEqual(res.shape, x.values.shape)
        for val_vector, res_vector in zip(x.values, res):
            scipy_res = special.softmax(val_vector)
            equal = np.allclose(res_vector, scipy_res, rtol=1e-6)
            self.assertTrue(equal)
            self.assertAlmostEqual(1., np.sum(res_vector))


if __name__ == '__main__':
    unittest.main()