import numpy as np
import unittest

def inverse_matrix_method(matrix_A, matrix_B):
    
    if matrix_A.shape[0] != matrix_A.shape[1]:
        raise ValueError("Matriks koefisien harus berbentuk persegi.")
    
   
    if matrix_A.shape[0] != matrix_B.shape[0]:
        raise ValueError("Jumlah baris matriks koefisien harus sama dengan jumlah baris vektor hasil.")
    
    
    A_inv = np.linalg.inv(matrix_A)
    

    solution = np.dot(A_inv, matrix_B)
    
    return solution

class TestInverseMatrixMethod(unittest.TestCase):
    def test_inverse_matrix_method(self):
        
        A = np.array([[3, 1], [2, -2]])
        B = np.array([7, -5])
        
        
        expected_solution = np.array([2, 1])
        
        
        actual_solution = inverse_matrix_method(A, B)
        
        
        np.testing.assert_array_almost_equal(actual_solution, expected_solution)

if __name__ == '__main__':
    unittest.main()
