from uf3.representation.knots import *


class TestKnots:
    def test_knot_sequence_from_points(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 3, 3, 3])

    def test_get_knot_subintervals(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        subintervals = get_knot_subintervals(sequence)
        assert np.allclose(subintervals[0], [1, 1, 1, 1, 2])
        assert np.allclose(subintervals[2], [1, 1, 2, 3, 3])
        assert np.allclose(subintervals[4], [2, 3, 3, 3, 3])

    def test_generate_uniform_knots(self):
        points = generate_uniform_knots(1, 6, 5, sequence=False)
        sequence = generate_uniform_knots(1, 6, 5, sequence=True)
        assert np.allclose(points, [1, 2, 3, 4, 5, 6])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    def test_lammps_knots(self):
        points = generate_lammps_knots(0, 1, 2)
        points = np.round(points, 4)
        assert np.allclose(points, [0, 0, 0, 0, 0.7071, 1, 1, 1, 1])
