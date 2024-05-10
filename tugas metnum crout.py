import numpy as np
import unittest

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1

    for k in range(n):
        U[k, k:] = A[k, k:] - np.dot(L[k, :k], U[:k, k:])
        L[k+1:, k] = (A[k+1:, k] - np.dot(L[k+1:, :k], U[:k, k])) / U[k, k]

    return L, U


# Contoh Implementasi
A = np.array([[3, -1, 5], [-2, 8, 4], [6, 3, 7]])
L, U = crout_decomposition(A)
print("Solusi SPL dengan Metode Dekomposisi Crout:")
print("L:", L)
print("U:", U)

class TestCroutDecomposition(unittest.TestCase):
    def test_decomposition(self):
        A = np.array([[3, -1, 5], [-2, 8, 4], [6, 3, 7]])
        expected_L = np.array([[1, 0, 0], [-0.66666667, 1, 0], [2, 0.5, 1]])
        expected_U = np.array([[3, -1, 5], [0, 8.66666667, 7.66666667], [0, 0, -3]])
        L, U = crout_decomposition(A)
        np.testing.assert_array_almost_equal(L, expected_L)
        np.testing.assert_array_almost_equal(U, expected_U)

if __name__ == '__main__':
    unittest.main()
