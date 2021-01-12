from uf3.regression.regularize import *


class TestRegularizers:
    def test_get_regularizer_matrix(self):
        m1 = get_regularizer_matrix(5, curvature=1, ridge=0)
        assert np.sum(m1) == 0
        assert np.sum(np.diag(m1)) == 8
        m2 = get_regularizer_matrix(5, curvature=2, ridge=0.5)
        assert np.sum(m2) == 2.5
        assert np.sum(np.diag(m2)) == 18.5

    def test_combine_regularizer_matrices(self):
        m1 = get_regularizer_matrix(4, curvature=1, ridge=0)
        m2 = get_regularizer_matrix(2, curvature=1, ridge=0)
        m3 = get_regularizer_matrix(3, curvature=1, ridge=0)
        matrix = combine_regularizer_matrices([m1, m2, m3])
        assert np.sum(matrix) == 0
        assert np.sum(np.diag(matrix)[:4]) == 6
        assert np.sum(np.diag(matrix)[4:6]) == 2
        assert np.sum(np.diag(matrix)[6:9]) == 4
