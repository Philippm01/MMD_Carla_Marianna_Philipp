import numpy as np
from scipy.sparse import csr_matrix
import unittest

def centered_cosine_sim(vec1, vec2):
 
    # Convert to dense arrays
    vec1_dense = vec1.toarray().flatten()
    vec2_dense = vec2.toarray().flatten()
    
    # Subtract means
    vec1_centered = vec1_dense - vec1_dense.mean()
    vec2_centered = vec2_dense - vec2_dense.mean()
    
    # Compute dot product and norms
    numerator = np.dot(vec1_centered, vec2_centered)
    denominator = np.linalg.norm(vec1_centered) * np.linalg.norm(vec2_centered)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def fast_centered_cosine_sim(matrix, vector):
  
    # Convert the vector to dense and subtract the mean
    vector_dense = vector.toarray().flatten()
    vector_centered = vector_dense - vector_dense.mean()
    
    # Initialize result array
    similarities = np.zeros(matrix.shape[0])
    
    # Loop over each row of the matrix
    for i in range(matrix.shape[0]):
        row_dense = matrix[i].toarray().flatten()
        row_centered = row_dense - row_dense.mean()
        
        # Compute dot product and norms
        numerator = np.dot(row_centered, vector_centered)
        denominator = np.linalg.norm(row_centered) * np.linalg.norm(vector_centered)
        
        if denominator == 0:
            similarities[i] = 0.0
        else:
            similarities[i] = numerator / denominator
    
    return similarities

class TestCenteredCosineSim(unittest.TestCase):

    def test_centered_cosine_sim_basic(self):
        """
        Test for k = 100, xi = i + 1
        """
        k = 100
        vector_x = np.array([i + 1 for i in range(k)])
        vector_y = np.flip(vector_x)
        
        vec_x_sparse = csr_matrix(vector_x)
        vec_y_sparse = csr_matrix(vector_y)
        
        result = centered_cosine_sim(vec_x_sparse, vec_y_sparse)
        print(f"Centered Cosine Similarity (basic): {result}")
        
        # Check that the similarity is approximately -1 (since vectors are reversed)
        self.assertAlmostEqual(result, -1.0, places=5)
    
    def test_centered_cosine_sim_with_nans(self):
        """
        Test for k = 100 with NaN values in specific indices.
        """
        k = 100
        vector_x = np.array([i + 1 for i in range(k)], dtype=float)
        vector_y = np.flip(vector_x)
        
        # Set NaN values for specific indices
        for c in [2, 3, 4, 5, 6]:
            for offset in range(0, 100, 10):
                idx = c + offset
                if idx < k:
                    vector_x[idx] = np.nan
        
        # Replace NaNs with zero in vector_x and vector_y for testing
        vec_x_sparse = csr_matrix(np.nan_to_num(vector_x, nan=0.0))
        vec_y_sparse = csr_matrix(np.nan_to_num(vector_y, nan=0.0))
        
        result = centered_cosine_sim(vec_x_sparse, vec_y_sparse)
        print(f"Centered Cosine Similarity (with NaNs): {result}")
        
        # Expect result not to be NaN and a valid similarity value
        self.assertFalse(np.isnan(result))
        self.assertTrue(-1 <= result <= 1)

# To run tests in Jupyter, we use the following approach
def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCenteredCosineSim)
    unittest.TextTestRunner(verbosity=2).run(suite)

# Run the tests
#run_tests()