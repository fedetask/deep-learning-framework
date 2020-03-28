import unittest
import numpy as np
from computation_graph import ComputationNode, Values, DotProduct, Sum, Softmax, Relu, Flatten
from computation_graph import Multiply, ReduceSum
from scipy import special


class ValuesTest(unittest.TestCase):

    def test_set_values_shape_none(self):
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


class DotProductTest(unittest.TestCase):

    def test_shape_first_None_vectors(self):
        x = Values(shape=(None, 12))  # Arbitrary number of vectors of shape (12,)
        y = Values(shape=(12,))  # One vector of shape (12,)
        dot = DotProduct()([x, y])
        self.assertEqual(dot.shape, (None, 1))

    def test_shape_both_None_vectors(self):
        x = Values(shape=(None, 12))  # Arbitrary number of vectors of shape (12,)
        y = Values(shape=(None, 12,))  # Arbitrary number of vectors of shape (12,)
        dot = DotProduct()([x, y])
        self.assertEqual(dot.shape, (None, 1))

    def test_shape_None_vector_by_matrix(self):
        x = Values(shape=(None, 12))  # Arbitrary number of vectors of shape (12,)
        y = Values(shape=(12, 7))  # A 12 x 7 matrix (e.g. a Dense layer with 7 units)
        dot = DotProduct()([x, y])
        self.assertEqual(dot.shape, (None, 7))

    def test_first_vector_second_None(self):
        x = Values(shape=(12,))  # One vector of shape (12,)
        y = Values(shape=(None, 12))  # Arbitrary number of vectors of shape (12,)
        dot = DotProduct()([x, y])
        self.assertEqual(dot.shape, (None, 1))

    def test_dotproduct_eval_vectors_by_matrix(self):
        x = Values(shape=(None, 15))
        x.set_values(np.random.normal(0, 1, size=(10, 15)))

        y = Values(shape=(15, 3))
        y.set_values(np.random.normal(0, 1, size=y.shape))

        z = DotProduct()([x, y])
        res = z.eval()
        expected = np.dot(x.values, y.values)
        self.assertTrue(np.array_equal(res, expected))


class SumTest(unittest.TestCase):

    def test_shape_fixed_vector(self):
        x = Values(shape=(10, 5))
        y = Values(shape=(10, 5))
        s = Sum()([x, y])
        self.assertEqual(s.shape, (10, 5))

    def test_shape_first_None_second_fixed_vectors(self):
        x = Values(shape=(None, 10, 5))
        y = Values(shape=(10, 5))
        s = Sum()([x, y])
        self.assertEqual(s.shape, (None, 10, 5))

    def test_shape_both_None_vectors(self):
        x = Values(shape=(None, 10, 5))
        y = Values(shape=(None, 10, 5))
        s = Sum()([x, y])
        self.assertEqual(s.shape, (None, 10, 5))

    def test_shape_first_fixed_second_None(self):
        x = Values(shape=(10, 5))
        y = Values(shape=(None, 10, 5))
        s = Sum()([x, y])
        self.assertEqual(s.shape, (None, 10, 5))

    def test_eval_fixed_shapes(self):
        x = Values(shape=(3, 5))
        x.set_values(np.array([[1, 2, 3, 4, 5],
                               [0, 1, 0, 1, 0],
                               [-1, 0, 2, 4, 6]]))
        y = Values(shape=(3, 5))
        y.set_values(np.ones((3,5)))
        s = Sum()([x, y])
        expected = np.array([[2, 3, 4, 5, 6],
                             [1, 2, 1, 2, 1],
                             [0, 1, 3, 5, 7]])
        equal = np.array_equal(expected, s.eval())
        self.assertTrue(equal)

    def test_eval_None_shapes(self):
        x = Values(shape=(None, 3, 5))
        x.set_values(np.array([
            [
                [1, 2, 3, 4, 5],
                [0, 1, 0, 1, 0],
                [-1, 0, 2, 4, 6]
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 4],
                [-1, 0, 7, 2, 6]
            ],
        ]))
        y = Values(shape=(3, 5))
        y.set_values(np.ones((3, 5)))
        s = Sum()([x, y])
        expected = np.array([
            [
                [2, 3, 4, 5, 6],
                [1, 2, 1, 2, 1],
                [0, 1, 3, 5, 7]
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 2, 1, 2, 5],
                [0, 1, 8, 3, 7]
            ],
        ])
        equal = np.array_equal(expected, s.eval())
        self.assertTrue(equal)


class FlattenTest(unittest.TestCase):

    def test_flatten_output_shape_when_parent_shape_none(self):
        x = Values(shape=(None, 10, 10, 3))  # Some 10 x 10 x 3 inputs
        x.set_values(np.ones(shape=(100, 10, 10, 3)))  # 100 inputs of shape 10 x 10 x 3

        flatten = Flatten()(x)
        self.assertEqual(flatten.shape, (None, 300))

    def test_flatten_output_shape_when_parent_shape_fixes(self):
        x = Values(shape=(10, 10, 3))  # Single input of shape 10 x 10 x 3
        x.set_values(np.ones(shape=(10, 10, 3)))

        flatten = Flatten()(x)
        self.assertEqual(flatten.shape, (1, 300))

    def test_flatten_eval_when_parent_shape_none(self):
        x = Values(shape=(None, 10, 10, 3))  # Some 10 x 10 x 3 inputs
        x.set_values(np.ones(shape=(200, 10, 10, 3)))  # 200 inputs

        flatten = Flatten()(x)
        res = flatten.eval()
        self.assertEqual(res.shape, (200, 300))


class SoftmaxTest(unittest.TestCase):

    def test_softmax_when_parent_shape_none_vectors(self):
        x = Values(shape=(None, 5))  # Any given number of 1x5 vectors
        vals = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [1, 1, 1, 1, 1]
        ])
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
        x = Values(shape=(3, 5))  # Any given number of 1x5 vectors
        vals = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [1, 1, 1, 1, 1]
        ])
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


class MultiplyTest(unittest.TestCase):

    def test_shape_scalar_by_batch(self):
        x = Values(shape=(None, 10))
        y = Values(shape=(1, ))
        m = Multiply()([x, y])
        self.assertEqual(m.shape, (None, 10))

    def test_shape_broadcasting_vector(self):
        x = Values(shape=(None, 10, 5))  # Now this is a matrix
        y = Values(shape=(5,))
        m = Multiply()([x, y])
        self.assertEqual(m.shape, (None, 10, 5))

    def test_shape_element_wise(self):
        x = Values(shape=(None, 15))
        y = Values(shape=(None, 15))
        m = Multiply()([x, y])
        self.assertEqual(m.shape, (None, 15))

    def test_eval_scalar_by_batch(self):
        x = Values(shape=(None, 4))
        x.set_values(np.array([
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [-1, -2, -3, -4],
            [2, 2, 2, 2],
        ]))
        y = Values(shape=(1))
        y.set_values(np.array([2]))
        m = Multiply()([x, y])
        expected = np.array([
            [2, 4, 6, 8],
            [0, 0, 0, 0],
            [-2, -4, -6, -8],
            [4, 4, 4, 4],
        ])
        res = m.eval()
        equal = np.array_equal(res, expected)
        self.assertTrue(equal)
        self.assertEqual(res.shape, (4, 4))


class ReduceSumTest(unittest.TestCase):

    def test_shape_matrix(self):
        x = Values(shape=(10, 5))
        r = ReduceSum()(x)
        self.assertEqual(r.shape, (10,))

    def test_shape_array_of_matrices(self):
        x = Values(shape=(10, 3, 5))  # 10 matrices 3 x 5
        r = ReduceSum()(x)
        self.assertEqual(r.shape, (10, 3))

    def test_shape_None_array_of_matrices(self):
        x = Values(shape=(None, 3, 5))  # 10 matrices 3 x 5
        r = ReduceSum()(x)
        self.assertEqual(r.shape, (None, 3))

    def test_shape_None_array_of_vectors(self):
        x = Values(shape=(None, 15))  # 10 matrices 3 x 5
        r = ReduceSum()(x)
        self.assertEqual(r.shape, (None,))

    def test_eval_matrix_first_None(self):
        x = Values(shape=(None, 15))
        x.set_values(np.array([
            np.array(range(15)),
            np.ones(15)
        ]))
        r = ReduceSum()(x)
        expected = np.array([14 * 15 / 2, 15])
        actual = r.eval()
        equal = np.array_equal(expected ,actual)
        self.assertTrue(equal)
        self.assertEqual(actual.shape, (2, ))
        self.assertEqual(actual.shape[1:], r.shape[1:])

    def test_eval_array_of_matrices_first_None(self):
        x = Values(shape=(None, 2, 3))
        x.set_values(np.array([
            [[1, 2, 3], [4, 5, 6]],  # A 2 x 3 matrix
            np.ones((2, 3))  # A 2 x 3 matrix of ones
        ]))
        r = ReduceSum()(x)
        expected = np.array([
            [6, 15],
            [3, 3]
        ])
        actual = r.eval()
        equal = np.array_equal(expected ,actual)
        self.assertTrue(equal)
        self.assertEqual(actual.shape, (2, 2))
        self.assertEqual(actual.shape[1:], r.shape[1:])


if __name__ == '__main__':
    unittest.main()
